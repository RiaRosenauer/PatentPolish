import base64
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import difflib
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field
from patentpolish.patent_model import Patent

from typing import Optional, Literal


import sys
import os

from patentpolish.api_connector import ApiConnector
from patentpolish.util import logger

try:
    # Cannot use local import
    # https://github.com/ronaldoussoren/pyobjc/issues/619
    import Cocoa  # type: ignore
    import objc  # type: ignore
    import Vision  # type: ignore
except ModuleNotFoundError:
    pass

class Textbox(BaseModel):
    text: str
    x: float
    y: float
    w: float
    h: float
    confidence: float



class ImageModel(Enum):
  DryRun = "DryRun"
  OpenAI = "OpenAI"
  Tesseract = "Tesseract"
  MacOS = "MacOS"

class TextModel(Enum):
  DryRun = "DryRun"
  OpenAI = "OpenAI"

class ImageReferenceSignOccurency(BaseModel):
  sign: str = Field(title='Reference sign', description='The number or letters used for referencing the part of the figure.')
  figure: int = Field(title='Figure number', description='The number of the figure in which the reference sign occured.')

  def __str__(self):
    return f"Sign {self.sign} in figure {self.figure}"


class ImageReferenceSigns(BaseModel):
  reference_sign_occurencies: list[ImageReferenceSignOccurency] = Field(title='Reference sign occurencies', description='Exhaustive list of reference signs occuring in the figures.')
  figure_to_page_mapping: list[int] = Field(title='Figure to Appendix mapping', description='List of Appendix numbers corresponding to the figure at the respective index.', default_factory=list)

  def __str__(self):
    return "\n".join([str(occ) for occ in self.reference_sign_occurencies])

class TextType(str, Enum):
  ABSTRACT = 'abstract'
  CLAIMS = 'claims'
  DESCRIPTION = 'description'

  def __str__(self):
    return self.value
  

class SimpleTextReferenceSignOccurency(BaseModel):
  sign: str = Field(title='Reference sign number', description='The number used for referencing the part of the figure.')
  name: str = Field(title=' Reference sign name', description='The concept the sign represents.')
  whole_sentence: str = Field(title='Whole sentence', description='The entire sentence where the sign occurs.')


class ListOfSimpleTextSigns(BaseModel):
  reference_sign_occurencies: list[SimpleTextReferenceSignOccurency] = Field(title='Reference sign occurencies', description='Exhaustive list of reference signs occuring in the abstract, claims and description.')


class TextReferenceSignOccurency(BaseModel):
  sign: str = Field(title='Reference sign number', description='The number used for referencing the part of the figure.')
  name: str = Field(title=' Reference sign name', description='The concept the sign represents.')
  whole_sentence: str = Field(title='Whole sentence', description='The entire sentence where the sign occurs.')
  text_type: TextType = Field(title='Text type', description='Whether the sign occured in the abstract, the claims or the description.')
  index_number: int = Field(default=None, title='Number of the description or claim', description='None if the occurrence is in the abstract, or the index (starting from 1) in the claims or description list.')

  @property
  def list_index(self):
    if getattr(self, "index_number", None) is None:
      return None
    return self.index_number - 1

  def __str__(self):
    # case switch
    type_str = {
      TextType.ABSTRACT: "Abstr.",
      TextType.CLAIMS: "Claims.",
      TextType.DESCRIPTION: "Desc."
    }

    if self.index_number is None:
      return f"Sign {self.sign} in {type_str[self.text_type]}"
    else:
      return f"Sign {self.sign} in {type_str[self.text_type]}{self.index_number}"


class TextReferenceSigns(BaseModel):
  reference_sign_occurencies: list[TextReferenceSignOccurency] = Field(title='Reference sign occurencies', description='Exhaustive list of reference signs occuring in the abstract, claims and description.')

  def __str__(self):
    return "\n".join([str(occ) for occ in self.reference_sign_occurencies])

class SignAnalysis(BaseModel):
  number: str
  text_occurencies: list[TextReferenceSignOccurency]
  figure_occurencies: list[ImageReferenceSignOccurency]
  concepts: dict[str, list[TextReferenceSignOccurency]]

  concepts_deviate: bool
  sign_only_in_text: bool
  sign_only_in_figure: bool

  def __str__(self):
    text_occ = [occ.sign for occ in self.text_occurencies]
    if self.is_in_both:
      return f"Sign {self.number} is in figures {self.figure_occurencies} and text {text_occ}"
    elif len(self.figure_occurencies) > 0:
      return f"Sign {self.number} is only in figures {self.figure_occurencies}"
    elif len(self.text_occurencies) > 0:
      return f"Sign {self.number} is only in text {text_occ}"
    else:
      return f"Sign {self.number} is not in any figures or text"
  
  @property
  def is_in_both(self):
    return len(self.figure_occurencies) > 0 and len(self.text_occurencies) > 0

  @property
  def no_error(self):
    no_error = True
    if len(self.text_occurencies) == 0 and len(self.figure_occurencies) == 0:
        no_error = False

    if self.sign_only_in_text:
        no_error = False

    if self.sign_only_in_figure:
        no_error = False

    if self.concepts_deviate:
        no_error = False

    return no_error


class Examiner:

  def __init__(self, patent: Patent = None, reference_number: str = None):
      self.client = OpenAI()
      if isinstance(patent, Patent):
          self.pat = patent
      elif reference_number:
          self.patent_number = reference_number
          connector = ApiConnector()
          self.pat = connector.get_patent(reference_number)
      elif patent:
          self.patent_number = patent
          connector = ApiConnector()
          self.pat = connector.get_patent(patent)
      else:
          raise ValueError("Either a patent object or a reference number must be provided.")
   

  # Function to encode the image
  @staticmethod
  def _encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

  def list_of_reference_signs(self):
    logger.critical("Sending Images to OpenAI")
    messages = [
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "You are a scrupulous patent examiner who never makes mistakes."
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Provide an exhaustive list of reference sings (meaning numbers or letters) includung the figure in which they occur in these technical drawings."
            },        
          ]
        }
    ]

    for i, image in enumerate(self.pat.image_paths):
      base64_image = self._encode_image(image)
      messages.append({
        "role": "user",
        "content": [
            {
              "type": "text",
              "text": f"Appendix number: {i+1}"
            },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      })

    completion_images = self.client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=messages,
        response_format=ImageReferenceSigns,
    )

    signs_images = completion_images.choices[0].message.parsed

    logger.critical("Sending texts to OpenAI")

    messages = [
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "You are a scrupulous patent examiner who never makes mistakes."
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Provide an exhaustive list of reference sings (meaning numbers or sometimes letters) with their name (the concept or object they represent), used in this abstract, claims and description."
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "ABSTRACT:\n"+self.pat.abstract
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "CLAIMS:\n"+"\n".join(self.pat.claims)
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "DESCRIPTION:\n"+"\n".join(self.pat.descriptions)
            },        
          ]
        }
    ]

    completion_text = self.client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=messages,
        response_format=TextReferenceSigns,
    )

    signs_text = completion_text.choices[0].message.parsed

    signs = set()
    for sign in signs_images:
      signs.add(sign.sign)
    for sign in signs_text:
      signs.add(sign.sign)

    return signs

    

  def oai_extract_signs_from_images(self) -> None:
    logger.critical("Sending Images to OpenAI")

    messages = [
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "You are a scrupulous patent examiner who never makes mistakes."
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Provide an exhaustive list of reference sings (meaning numbers) includung the figure in which they occur in these technical drawings."
            },        
          ]
        }
    ]

    for image in self.pat.image_paths:
      base64_image = self._encode_image(image)
      messages.append({
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      })

    completion = self.client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=messages,
        response_format=ImageReferenceSigns,
    )


    signs = completion.choices[0].message.parsed

    self.image_signs = signs


  def heavy_extract_signs_from_texts(self):
    logger.info("Using heavy extraction method for text signs")
    
    all_occurencies = []

    total_amount = 1 + len(self.pat.claims) + len(self.pat.descriptions)

    # Process abstract
    abstract_signs = self._extract_signs_from_text(self.pat.abstract, TextType.ABSTRACT)
    logger.info(f"Processing Abstract")
    yield "Processing Abstract", 0.0
    all_occurencies.extend([
        TextReferenceSignOccurency(sign=sign.sign, name=sign.name, whole_sentence=sign.whole_sentence, text_type=TextType.ABSTRACT)
        for sign in abstract_signs
    ])

    # Process claims
    logger.info(f"Processing Claims")
    for i, claim in enumerate(self.pat.claims):
        yield f"Processing Claim {i+1} out of {len(self.pat.claims)}", (i+1) /total_amount
        claim_signs = self._extract_signs_from_text(claim, TextType.CLAIMS)
        all_occurencies.extend([
            TextReferenceSignOccurency(sign=sign.sign, name=sign.name, whole_sentence=sign.whole_sentence, text_type=TextType.CLAIMS, index_number=i+1)
            for sign in claim_signs
        ])

    # Process descriptions
    for i, description in enumerate(self.pat.descriptions):
        yield f"Processing Description {i+1} out of {len(self.pat.descriptions)}", (len(self.pat.claims) + i + 1) /total_amount
        desc_signs = self._extract_signs_from_text(description, TextType.DESCRIPTION)
        all_occurencies.extend([
            TextReferenceSignOccurency(sign=sign.sign, name=sign.name, whole_sentence=sign.whole_sentence, text_type=TextType.DESCRIPTION, index_number=i+1)
            for sign in desc_signs
        ])

    self.text_signs = TextReferenceSigns(reference_sign_occurencies=all_occurencies)
    yield "Finished", 1.0

  def _extract_signs_from_text(self, text: str, text_type: TextType) -> list[SimpleTextReferenceSignOccurency]:
    
    
    messages = [
        {
            "role": "system",
            "content": "You are a scrupulous patent examiner who never makes mistakes."
        },
        {
            "role": "user",
            "content": f"""Provide an exhaustive list of reference signs (meaning numbers) with their name (the concept they represent), used in this {text_type.value}.
            Be careful to only include something if it actually makes sense in the sentence context as a reference sign. 
            Check the context of each occurrence if it could also be a figure number, a list number, the mention of a claim or something else. 
            Also if it is part of another reference sign (e.g. 2 is part of 12) then do not include it. Rather return an empty list than false positives.
            Examples that should NOT be included for reference sign 2:
            - figure 2 shows in more detail a perspective view of a part of the apparatus of figure 1 with the edge-forming means therein; (here 2 is meant as figure 2 and not a reference sign)
            - Figure 3 shows edge-forming means 22 in even more detail. (here 2 is part of 22 and not 2)
            - Figure 2 shows in more detail the part of apparatus 1 comprising edge-forming means 22. (here both edge cases are present)
            It is okay (and better) if you return an empty list if you think there are no occurrences."""
        },
        {
            "role": "user",
            "content": text
        }
    ]

    completion = self.client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=messages,
        response_format=ListOfSimpleTextSigns,
    )

    return completion.choices[0].message.parsed.reference_sign_occurencies

  def oai_extract_signs_from_texts(self):
    logger.critical("Sending texts to OpenAI")

    claim_string = "\n".join([f"CLAIM {i+1}:\n{claim}\n" for i, claim in enumerate(self.pat.claims)])
    description_string = "\n".join([f"DESCRIPTION {i+1}:\n{desc}\n" for i, desc in enumerate(self.pat.descriptions)])
    print("Claim string of length", len(claim_string))
    print("Description of length", len(description_string))


    if len(description_string) + len(claim_string) > 20000:
      print("Description is longer than 20000 characters, using heavy extraction method.")
      yield from self.heavy_extract_signs_from_texts()
    else:
      yield "Processing patent in one go", 0.0
      messages = [
          {
            "role": "system",
            "content": [
              {
                "type": "text",
                "text": "You are a scrupulous patent examiner who never makes mistakes."
              },        
            ]
          },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": """Provide an exhaustive list of reference sings (meaning numbers) with their name (the concept they represent), used in this abstract, claims and description.
                But be carefull, only include something if it 
                actually makes sense in the sentence context, that it is a reference sign. 
                Check the context of each occurence if it could also be a figure number, a list number, the mention of a claim or something else. 
                Also if it is part of another reference sign (e.g. 2 is part of 12) then do not include it. Rather return an empty list than false positives.
                Examples that should NOT be included for for example reference sign 2:
                figure 2 shows in more detail a perspective view of a part of the apparatus of figure 1 with the edge-forming means therein; (here 2 is meant as figure 2 and not a reference sign)
                Figure 3 shows edge-forming means 22 in even more detail. (here 2 is part of 22 and not 2)
                Figure 2 shows in more detail the part of apparatus 1 comprising edge-forming means 22. (here both edge cases are present)
                It is okay (and better) if you return an empty list if you think there are no occurences."""
              },        
            ]
          },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "ABSTRACT:\n"+self.pat.abstract
              },        
            ]
          },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": claim_string
              },        
            ]
          },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": description_string
              },        
            ]
          }
      ]

      completion = self.client.beta.chat.completions.parse(
          model="gpt-4o-2024-08-06",
          temperature=0,
          messages=messages,
          response_format=TextReferenceSigns,
      )

      signs = completion.choices[0].message.parsed

      self.text_signs = signs
    yield "finished", 1.0

  def oai_check_text_not_occurence(self, sign)  -> TextReferenceSigns:
    messages = [
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "You are a scrupulous patent examiner who never makes mistakes."
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"""Provide an exhaustive list of the text occurencies of this text reference sign: {sign}. But be carefull, only include something if it 
              actually makes sense in the sentence context, that it is a reference sign. 
              Check the context of each occurence if it could also be a figure number, a list number, the mention of a claim or something else. 
              Also if it is part of another reference sign (e.g. 2 is part of 12) then do not include it. Rather return an empty list than false positives.
              Examples that should NOT be included in the returned list for for example reference sign 2:
              figure 2 shows in more detail a perspective view of a part of the apparatus of figure 1 with the edge-forming means therein; (here 2 is meant as figure 2 and not a reference sign)
              Figure 3 shows edge-forming means 22 in even more detail. (here 2 is part of 22 and not 2)
              Figure 2 shows in more detail the part of apparatus 1 comprising edge-forming means 22. (here both edge cases are present)
              It is okay (and better) if you return an empty list if you think there are no occurences."""
            },        
          ]
        }, 
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "ABSTRACT:\n"+self.pat.abstract
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "\n".join([f"CLAIMS {i+1}:\n{claim}\n" for i, claim in enumerate(self.pat.claims)])
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "\n".join([f"DESCRIPTION {i+1}:\n{desc}\n" for i, desc in enumerate(self.pat.descriptions)])
            },        
          ]
        }
    ]

    completion = self.client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=messages,
        response_format=TextReferenceSigns,
    )

    signs = completion.choices[0].message.parsed

    return signs


  def oai_check_text_not_occurence(self, sign)  -> TextReferenceSigns:
    messages = [
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "You are a scrupulous patent examiner who never makes mistakes."
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"""Provide an exhaustive list of the text occurencies of this text reference sign: {sign}. But be carefull, only include something if it 
              actually makes sense in the sentence context, that it is a reference sign. 
              Check the context of each occurence if it could also be a figure number, a list number, the mention of a claim or something else. 
              Also if it is part of another reference sign (e.g. 2 is part of 12) then do not include it. Rather return an empty list than false positives.
              Examples that should NOT be included in the returned list for for example reference sign 2:
              figure 2 shows in more detail a perspective view of a part of the apparatus of figure 1 with the edge-forming means therein; (here 2 is meant as figure 2 and not a reference sign)
              Figure 3 shows edge-forming means 22 in even more detail. (here 2 is part of 22 and not 2)
              Figure 2 shows in more detail the part of apparatus 1 comprising edge-forming means 22. (here both edge cases are present)
              It is okay (and better) if you return an empty list if you think there are no occurences."""
            },        
          ]
        }, 
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "ABSTRACT:\n"+self.pat.abstract
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "CLAIMS:\n"+"\n".join(self.pat.claims)
            },        
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "DESCRIPTION:\n"+"\n".join(self.pat.descriptions)
            },        
          ]
        }
    ]

    completion = self.client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=messages,
        response_format=TextReferenceSigns,
    )

    signs = completion.choices[0].message.parsed

    return signs


  def regex_extract_signs(self):
    big_ol_string = self.pat.abstract + " ".join(self.pat.claims) + " ".join(self.pat.descriptions)
    signs = re.findall(r"\((\d+)\)", big_ol_string)
    self.regex_signs = list(set(map(int, signs)))

  def cheap_is_in_concepts(self, concept, concepts):
    return any(self.cheap_similar_concept(concept, other) for other in concepts)

  def cheap_similar_concept(self, concept1, concept2):
    if concept1 == concept2:
      return True
    if concept1 in concept2 or concept2 in concept1:
      return True
    
    s = difflib.SequenceMatcher(None, concept1, concept2)
    _, _, size = s.find_longest_match(0, len(concept1), 0, len(concept2))
    if size/max(len(concept1), len(concept2)) > 0.7:
      return True

  def similar_concept(self, concept1, concept2):
    if self.cheap_similar_concept(concept1, concept2):
      return True

    
    class ConceptuallySimilar(BaseModel):
      similar: bool = Field(title='ConceptuallySimilar', description='Whether or not the two provided descriptions might describe the same thing.')
    
    completion = self.client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=[
          {
            "role": "system",
            "content": [
              {
                "type": "text",
                "text": "You are a linguist who is very good at understanding the meaning of words."
              },        
            ]
          },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": f"Is it possible that '{concept1}' and '{concept2}' describe the same thing?"
              },        
            ]
          }
        ],
        response_format=ConceptuallySimilar,
    )

    return completion.choices[0].message.parsed.similar

  def compare_signs(self):
    if not hasattr(self, "text_signs") or not hasattr(self, "image_signs"):
      raise ValueError("You need to extract the signs from the texts and images first.")

    text_signs = set([sign.sign for sign in self.text_signs.reference_sign_occurencies])
    image_signs = set([sign.sign for sign in self.image_signs.reference_sign_occurencies])

    print(text_signs)
    print(image_signs)

    all_signs = sorted(text_signs.union(image_signs))
    for sign in all_signs:
      text_occurencies = []
      figure_occurencies = []
      name = None

      if sign in text_signs:
        text_occurencies = [occ.text_type for occ in self.text_signs.reference_sign_occurencies if occ.sign == sign]
        names = [occ.name for occ in self.text_signs.reference_sign_occurencies if occ.sign == sign]
        name = names[0]

        all_same = True
        for i, name1 in enumerate(names):
          for name2 in names[i+1:]:
            if not self.similar_concept(name1, name2):
              all_same = False
              logger.warning(f"Concepts deviating for sign {sign}: '{name1}' vs '{name2}'")
          if all_same:
            if len(names) > 1:
              logger.info(f"Concepts similar for {sign}: {names}")
            break

      if sign in image_signs:
        figure_occurencies = [occ.figure for occ in self.image_signs.reference_sign_occurencies if occ.sign == sign]

      yield SignAnalysis(number=sign, name=name, figure_occurencies=figure_occurencies, text_occurencies=text_occurencies)

  def compare_signs_frontend(self):
    if not hasattr(self, "text_signs") or not hasattr(self, "image_signs"):
      raise ValueError("You need to extract the signs from the texts and images first.")

    text_signs = set([sign.sign for sign in self.text_signs.reference_sign_occurencies])
    image_signs = set([sign.sign for sign in self.image_signs.reference_sign_occurencies])

    all_signs = sorted(text_signs.union(image_signs))
    sign_list = []

    for sign in all_signs:
      text_occurencies = []
      figure_occurencies = []
      concepts = {}

      if sign not in text_signs:
        returned_list = self.oai_check_text_not_occurence(sign)
        text_occurencies = [occ for occ in returned_list.reference_sign_occurencies if occ.sign == sign]

      if sign in text_signs or len(text_occurencies) > 0:
        if len(text_occurencies) == 0:
          text_occurencies = [occ for occ in self.text_signs.reference_sign_occurencies if occ.sign == sign]
        
        all_same = True
        if len(text_occurencies) == 1:
          occ = text_occurencies[0]
          concepts[occ.name] = [occ]

        for i in range(len(text_occurencies) - 1):
          occ1 = text_occurencies[i]
          occ2 = text_occurencies[i + 1]
          
          # Check if the two occurrences should be treated similarly
          if self.similar_concept(occ1.name, occ2.name):
            # If they are similar, treat them as the same concept
            if occ1.name not in concepts:
              concepts[occ1.name] = [occ1]  # Initialize the list for this concept if it doesn't exist
              print('concept after adding occurence name 1: ', concepts)

            if occ2 not in concepts[occ1.name]:
              concepts[occ1.name].append(occ2)
              
          else:
            # If concepts deviate, treat them separately
            if not self.cheap_is_in_concepts(occ1.name, concepts):
              print("in adding occurence name 1")
              concepts[occ1.name] = [occ1]
              print('concept after adding occurence name 1: ', concepts)


            if not self.cheap_is_in_concepts(occ2.name, concepts):
              print("in adding occurence name 2")
              concepts[occ2.name] = [occ2]

            all_same = False
            logger.warning(f"Concepts deviating for sign {sign}: '{occ1.name}' vs '{occ2.name}'")

        if all_same:
            if len(concepts) > 1:
                logger.info(f"Concepts similar for sign {sign}: {concepts}")


      if sign in image_signs:
        figure_occurencies = [occ for occ in self.image_signs.reference_sign_occurencies if occ.sign == sign]

      # Correcting the conditional statements
      sign_only_in_text = len(text_occurencies) > 0 and len(figure_occurencies) == 0
      sign_only_in_figure = len(figure_occurencies) > 0 and len(text_occurencies) == 0

      # Creating an analysed_sign object
      analysed_sign = SignAnalysis(
          number=sign, 
          text_occurencies=text_occurencies,
          figure_occurencies=figure_occurencies, 
          concepts=concepts, 
          concepts_deviate=not all_same,
          sign_only_in_text=sign_only_in_text,
          sign_only_in_figure=sign_only_in_figure
      )
      # Appending the analysed sign to the sign_list
      sign_list.append(analysed_sign)

    return sign_list


def run_comparison(patent_reference_number: str, image_model: ImageModel = ImageModel.DryRun, text_model:TextModel = TextModel.DryRun):

  examiner = Examiner(patent_reference_number)

  examiner.image_signs = ImageReferenceSigns(reference_sign_occurencies=[
      ImageReferenceSignOccurency(sign="1", figure=1),
      ImageReferenceSignOccurency(sign="3", figure=1),
      ImageReferenceSignOccurency(sign="4", figure=1),
      ImageReferenceSignOccurency(sign="5", figure=2),
    ])

  examiner.text_signs = TextReferenceSigns(reference_sign_occurencies=[
      TextReferenceSignOccurency(sign="1", name="foo", text_type=TextType.ABSTRACT, whole_sentence="foo", index_number=1),
      TextReferenceSignOccurency(sign="2", name="bar", text_type=TextType.CLAIMS, whole_sentence="bar", index_number=2),
      TextReferenceSignOccurency(sign="4", name="means for carring take-off plates", text_type=TextType.DESCRIPTION, whole_sentence="means for carring take-off plates", index_number=4),
      TextReferenceSignOccurency(sign="4", name="take-off plate transportation device", text_type=TextType.DESCRIPTION, whole_sentence="take-off plate transportation device", index_number=4),
      TextReferenceSignOccurency(sign="5", name="means for moving the mould container parts", text_type=TextType.DESCRIPTION, whole_sentence="some sentence"),
      TextReferenceSignOccurency(sign="5", name="edge-forming means", text_type=TextType.DESCRIPTION, whole_sentence="some sentence"),
    ])

  if image_model == ImageModel.OpenAI:
    examiner.oai_extract_signs_from_images()

  if text_model == TextModel.OpenAI:
    examiner.oai_extract_signs_from_texts()

  print("\nComparative analysis:")
  for sign in examiner.compare_signs_frontend():
    if sign.is_in_both:
        logger.info(sign)
    else:
        logger.error(sign)

if __name__ == "__main__":
  patent_reference_number = "EP4353111"
  examiner = Examiner(patent_reference_number)

  examiner.image_signs = ImageReferenceSigns(reference_sign_occurencies=[
      ImageReferenceSignOccurency(sign="1", figure=1),
      ImageReferenceSignOccurency(sign="3", figure=1),
      ImageReferenceSignOccurency(sign="4", figure=1),
      ImageReferenceSignOccurency(sign="5", figure=2),
    ])

  examiner.text_signs = TextReferenceSigns(reference_sign_occurencies=[
      TextReferenceSignOccurency(sign="1", name="foo", text_type=TextType.ABSTRACT, whole_sentence="foo", index_number=1),
      TextReferenceSignOccurency(sign="2", name="bar", text_type=TextType.CLAIMS, whole_sentence="bar", index_number=2),
      TextReferenceSignOccurency(sign="4", name="means for carring take-off plates", text_type=TextType.DESCRIPTION, whole_sentence="means for carring take-off plates", index_number=4),
      TextReferenceSignOccurency(sign="4", name="take-off plate transportation device", text_type=TextType.DESCRIPTION, whole_sentence="take-off plate transportation device", index_number=4),
      TextReferenceSignOccurency(sign="5", name="means for moving the mould container parts", text_type=TextType.DESCRIPTION, whole_sentence="some sentence"),
      TextReferenceSignOccurency(sign="5", name="edge-forming means", text_type=TextType.DESCRIPTION, whole_sentence="some sentence"),
    ])

  
  for status, progress in examiner.oai_extract_signs_from_texts():
    logger.info(f"{status}: {progress}")

