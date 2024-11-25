import csv
from dataclasses import dataclass

from patentpolish.sign_examiner import (
  Examiner,
  ImageReferenceSignOccurency,
  ImageReferenceSigns,
  TextReferenceSignOccurency,
  TextReferenceSigns,
  TextType,
)
from patentpolish.util import logger, DATA_DIR


@dataclass
class SignAnalysis:
  number: int
  name: str
  figure_occurencies: list[int]
  text_occurencies: list[TextType]

  def __str__(self):
    text_occ = [occ.value for occ in self.text_occurencies]
    if self.is_in_both:
      return f"Sign {self.number} ({self.name}) is in figures {self.figure_occurencies} and text {text_occ}"
    elif len(self.figure_occurencies) > 0:
      return f"Sign {self.number} is only in figures {self.figure_occurencies}"
    elif len(self.text_occurencies) > 0:
      return f"Sign {self.number} ({self.name}) is only in texts {text_occ}"
    else:
      return f"Sign {self.number} ({self.name}) is not in any figures or text"
  
  @property
  def is_in_both(self):
    return len(self.figure_occurencies) > 0 and len(self.text_occurencies) > 0


class ExpectationJudge:
    def __init__(self, sign_examiner: Examiner) -> None:
        self.examiner = sign_examiner
    
    def get_image_expectations(self):
        occs = []
        with open(DATA_DIR+f"/expectations/{self.examiner.patent_number}_img.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                occs.append(ImageReferenceSignOccurency(sign=row[0].strip(), figure=int(row[1].strip())))
        
        
        return ImageReferenceSigns(reference_sign_occurencies=occs)

    def check_image_expectations(self):
        if not hasattr(self.examiner, "image_signs"):
            raise ValueError("You need to extract the signs from the images first.")


        ex_tuples = [(sign.sign, sign.figure) for sign in self.get_image_expectations().reference_sign_occurencies]
        obs_tuples = [(sign.sign, sign.figure) for sign in self.examiner.image_signs.reference_sign_occurencies]
        union = sorted(list(set(ex_tuples) | set(obs_tuples)), key=lambda x: (x[0].zfill(3) if x[0].isdigit() else x[0], x[1]))


        for tup in union:
            occurency = ImageReferenceSignOccurency(sign=tup[0],figure=tup[1])
            if tup in ex_tuples and tup in obs_tuples:
                logger.info(f"Found expected {occurency}")
            elif tup in obs_tuples:
                logger.error(f"Unexpected {occurency}")
            elif tup in ex_tuples:
                logger.error(f"Missing {occurency}")
            else:
                logger.error(f"{tup} not expected and not found") # should not happen, right?   

    def get_text_expectations(self):
        # read in data/expect_txts/EP.1000000.A1.csv
        occs = []
        with open(DATA_DIR+f"/expectations/{self.examiner.patent_number}_txt.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                typ = TextType(row[1].strip().lower())
                if typ == TextType.ABSTRACT:
                    occ = TextReferenceSignOccurency(sign=row[0].strip(), text_type=typ, name="", whole_sentence="")
                else:
                    occ = TextReferenceSignOccurency(sign=row[0].strip(), text_type=typ, name="", whole_sentence="", index_number=row[2].strip())
                
                occs.append(occ)

        return TextReferenceSigns(reference_sign_occurencies=occs)
        
    def check_text_expectations(self):
        if not hasattr(self.examiner, "text_signs"):
            raise ValueError("You need to extract the signs from the text first.")


        ex_tuples = [(sign.sign, sign.text_type, sign.index_number) for sign in self.get_text_expectations().reference_sign_occurencies]
        obs_tuples = [(sign.sign, sign.text_type, sign.index_number) for sign in self.examiner.text_signs.reference_sign_occurencies]
        union = sorted(list(set(ex_tuples) | set(obs_tuples)), key=lambda x: (str(x[0]).zfill(3), x[1], str(x[2]).zfill(3)))

        for sign in union:
            occurency = TextReferenceSignOccurency(name="", sign=sign[0],text_type=sign[1], whole_sentence="", index_number=sign[2])
            if sign in ex_tuples and sign in obs_tuples:
                logger.info(f"Found expected {occurency}")
            elif sign in obs_tuples:
                logger.error(f"Unexpected    {occurency}")
            elif sign in ex_tuples:
                logger.error(f"Missing       {occurency}")
            else:
                logger.error(f"{sign} not expected and not found") # should not happen, right?


if __name__ == "__main__":
    examiner = Examiner("EP.1000000.A1")
    judge = ExpectationJudge(examiner)

    check_images = False
    check_texts = False   
    
    check_images = True
    check_texts = True


    if check_images:
        judge.examiner.image_signs = judge.get_image_expectations()
        judge.examiner.oai_extract_signs_from_images() 
        print("\nPerformance on images:")
        judge.check_image_expectations()

    if check_texts:
        judge.examiner.text_signs = judge.get_text_expectations()
        judge.examiner.oai_extract_signs_from_texts()
        print("\nPerformance on text:")
        judge.check_text_expectations()

    
