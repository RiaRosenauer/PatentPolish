import base64
import time
from loguru import logger
import pandas as pd
import streamlit as st
from patentpolish.sign_examiner import Examiner, ImageModel, TextModel, TextType
from patentpolish.patent_model import Patent
from patentpolish.api_connector import ApiConnector
from patentpolish.util import split_into_sentences

from typing import List
from pydantic import BaseModel
import os
from pathlib import Path
import difflib
import pickle
import re
import pandas as pd
import csv

from patentpolish.util import DATA_DIR 

# Folder where images are stored
image_folder = Path(DATA_DIR) / "images"

patent_folder = Path(DATA_DIR) / "patents"
# Set the page layout to be wide to span the full display
st.set_page_config(layout="wide", page_icon=":pencil:", page_title="PatentPolish")

print("running update")

# Initialize session state for dynamic claims, descriptions, and images if not present
if 'claims' not in st.session_state:
    st.session_state.claims = []

if 'descriptions' not in st.session_state:
    st.session_state.descriptions = []

if 'images' not in st.session_state:
    st.session_state.images = []

if 'start_image' not in st.session_state:
    st.session_state.start_image = 0

if "check_formality" not in st.session_state:
    st.session_state.check_formality = False

if "check_formality_run" not in st.session_state:
    st.session_state.check_formality = False

if "patent" not in st.session_state:
    st.session_state.patent = None

if "patent_name" not in st.session_state:
    st.session_state.patent_name = None

# Function to add a new claim field
def add_claim():
    st.session_state.claims.append(f"Claim {len(st.session_state.claims) + 1}")

# Function to add a new description field
def add_description():
    st.session_state.descriptions.append(f"Description {len(st.session_state.descriptions) + 1}")

# Function to add a new image field
def add_image(image_path):
    st.session_state.images.append(image_path)

# Function to delete a description
def delete_description(index):
    st.session_state.descriptions.pop(index)

# Function to delete a claim
def delete_claim(index):
    st.session_state.claims.pop(index)

# Function to delete an image
def delete_image(index):
    st.session_state.images.pop(index)


# Custom function to handle non-integer reference signs
def convert_to_int_if_possible(value):
    try:
        # Try to convert the value to an integer
        return int(value)
    except ValueError:
        # If conversion fails, return the original value (likely a non-numeric string)
        return value

def list_to_string(list):
    if len(list) == 0:
        return "None"
    elif len(list) == 1:
        return list[0]
    else:
        before_and = ", ".join(str(item) for item in list[:-1])
        last = str(list[-1])
        return f"{before_and} and {last}"

def generate_sign_csv(sign_list, file_path):
    # Define the header for the CSV file
    csv_header = ["Reference sign", "Found in figures", "Conceptual meaning", "Sentence in text"]

    # Open the file in write mode
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(csv_header)

        # Write the data for each sign
        for sign in sign_list:
            # Gather the data for each sign
            sign_number = sign.number

            # Found in figures - This stays the same across all text occurrences
            figures = ", ".join([str(occ.figure) for occ in sign.figure_occurencies]) if sign.figure_occurencies else "NONE"

            # Write a separate row for each text occurrence
            if sign.text_occurencies:
                for occ in sign.text_occurencies:
                    # Extract the specific conceptual meaning for this text occurrence
                    concept_meaning = None
                    for concept, occurrences in sign.concepts.items():
                        if occ in occurrences:
                            concept_meaning = concept
                            break

                    # Fallback to "NONE" if no conceptual meaning found
                    concept_meaning = concept_meaning if concept_meaning else "NONE"

                    sentence = occ.whole_sentence
                    writer.writerow([sign_number, figures, concept_meaning, sentence])
            else:
                # If no text occurrences, write a row with "NONE" for the sentence in text
                writer.writerow([sign_number, figures, "NONE", "NONE"])


if not getattr(st.session_state, "evaluation_stage", False):
    st.title("PatentPolish")

    

    st.write("Please enter a name for your new patent. Once you have saved the patent, it can loaded with the same name. If you enter a publication number of a already published patent, the form will be prepopulated with data from the Open Patent Services.")
    st.info("""Theoretically, you can run any patent from the EPO patent database. However, we have not extensively tested our 
    tool with a wide range of patents, so there is a possibility that the patent you are trying to analyze may not 
    work due to unexpected file formats, for example. The following patents have been tested and are confirmed to work: 
    EP.2000022, EP.1000000. We have tested random other patents, and most work, but please do not be disappointed if 
    your specific patent is not yet supported. This is a work in progress.""", icon="ℹ️")

    # Text input for the patent name
    st.session_state.patent_name = st.text_input("Patent ID", value="EP.2000022")
    patent_name = st.session_state.patent_name


    # Button to load the patent
    if st.button("Load/Create patent"):
        path = os.path.join(DATA_DIR, "patents", st.session_state.patent_name + ".json")

        # Check if the JSON file exists
        if os.path.exists(path):
            with open(path) as f:
                st.session_state.patent = Patent.model_validate_json(f.read())
                st.toast(f"Patent {st.session_state.patent_name} loaded from file.")
        
        if st.session_state.patent == None:
            api_connector = ApiConnector()

            try:
                st.session_state.patent = api_connector.get_patent(st.session_state.patent_name)
            except Exception as e:
                st.error(f"Error while loading patent. Try with a different patent number.\n Error: {e}")
                st.session_state.patent = None

            if st.session_state.patent != None:
                st.toast(f"Patent {st.session_state.patent_name} downloaded from EPO.")

        if st.session_state.patent == None:
            st.session_state.patent = Patent(reference_number=patent_name, title="", abstract="", claims=[], descriptions=[], image_paths=[], pdf_path="")

    if st.session_state.patent is not None:
        # Prefill fields from the loaded JSON
        patent_title = st.text_input("Enter Patent Title", value=st.session_state.patent.title, key="patent_title")

        uploaded_abstract = st.text_area("Enter Abstract", value=st.session_state.patent.abstract)

        with st.expander("Claims"):
            #Prefill fields from the loaded JSON only if the session state is empty
            st.session_state.claims = st.session_state.patent.claims
            
            # Create a container to hold the claims
            claims_container = st.container()

            # Display current claims in the container
            with claims_container:
                st.write("Claims:")
                for i, claim in enumerate(st.session_state.claims):
                    st.session_state.claims[i] = st.text_area(f"Claim {i+1}", value=claim, key=f"claim_{i}")

                    # Add a delete button for each description
                    if st.button(f"Delete Claim {i+1}", key=f"delete_claim{i}"):
                        delete_claim(i)
                        st.rerun()


            # Button to add more claims
            if st.button("Add more claims"):
                add_claim()
                # Add an empty text area for the new claim in the container
                with claims_container:
                    st.text_area(f"Claim {len(st.session_state.claims)}", value=f"Claim {len(st.session_state.claims)}", key=f"claim_{len(st.session_state.claims)-1}")

                    if st.button(f"Delete Claim {len(st.session_state.claims)}", key=f"delete_claim{len(st.session_state.claims)}"):
                        delete_claim(len(st.session_state.claims)-1)
                        st.rerun()


        with st.expander("Descriptions"):
            #Prefill fields from the loaded JSON only if the session state is empty
            st.session_state.descriptions = st.session_state.patent.descriptions
            
            # Create a container to hold the descriptions
            descriptions_container = st.container()

            with descriptions_container:
                st.write("Descriptions:")
                for i, description in enumerate(st.session_state.descriptions):
                    st.session_state.descriptions[i] = st.text_area(f"Description {i+1}", value=description, key=f"description_{i}")

                    # Add a delete button for each description
                    if st.button(f"Delete Description {i+1}", key=f"delete_description_{i}"):
                        delete_description(i)
                        st.rerun()

            if st.button("Add more descriptions"):
                add_description()
                with descriptions_container:
                    st.text_area(f"Description {len(st.session_state.descriptions)}", value=f"Description {len(st.session_state.descriptions)}", key=f"description_{len(st.session_state.descriptions)-1}")
                    # Add a delete button for each description
                    if st.button(f"Delete Description {len(st.session_state.descriptions)+1}", key=f"delete_description_{len(st.session_state.descriptions)}"):
                        delete_description(len(st.session_state.descriptions))
                        st.rerun()

        with st.expander("Images"):
            # Display the images
            st.session_state.images = st.session_state.patent.image_paths
            
            images_container = st.container()

            with images_container:
                st.write("Images:")
                for i, image_path in enumerate(st.session_state.images):
                    st.image(image_path, caption=f"Image {i+1}", use_column_width=True)

                    if st.button(f"Delete Image {i+1}", key=f"delete_image_{i}"):
                        delete_image(i)
                        st.rerun()

            # Allow users to upload images
            uploaded_images = st.file_uploader("Upload new images (jpg or png)", type=["jpg", "png"], accept_multiple_files=True)

            if uploaded_images:
                # Count the number of already existing images for the patent
                existing_images = [img for img in os.listdir(image_folder) if img.startswith(f"{patent_name}_")]
                image_count = len(existing_images)  # To ensure image numbering is sequential

                # Process each uploaded image
                for i, uploaded_image in enumerate(uploaded_images, start=1):
                    if i > st.session_state.start_image:
                        # Rename the image with patent name and sequential number
                        image_number = image_count + i-st.session_state.start_image

                        new_image_name = f"{patent_name}_{image_number:02}.png"  # Change extension if needed
                        image_path = os.path.join(image_folder, new_image_name)

                        # Save the image with the new name
                        with open(image_path, "wb") as f:
                            f.write(uploaded_image.getbuffer())

                        # Add the image path to session state
                        add_image(image_path)

                        # Display the image with the new name in the container
                        with images_container:
                            st.image(image_path, caption=f"Image {image_number}", use_column_width=True)
                            if st.button(f"Delete Image {image_number}", key=f"delete_image_{image_number-1}"):
                                delete_image(image_number-1)
                                st.rerun()

                st.session_state.start_image = i


        col1, col2 = st.columns([1,4])
        with col1:
            check_formality = st.button("Check formality")
        with col2:
            save_changes = st.button("Save changes")

        if save_changes:

            # Create a Patent object with the updated data
            st.session_state.updated_patent = Patent(
                reference_number=patent_name,
                title=patent_title,
                abstract=uploaded_abstract,
                claims=[claim for claim in st.session_state.claims],
                descriptions=[description for description in st.session_state.descriptions],
                image_paths=[image_path for image_path in st.session_state.images],
                pdf_path=st.session_state.patent.pdf_path
            )

            # Save the updated patent information as a JSON file
            json_path = os.path.join(patent_folder, f"{patent_name}.json")
            with open(json_path, "w") as f:
                f.write(st.session_state.updated_patent.model_dump_json())
            
            # Display success message
            st.success(f"Patent {patent_name} has been updated successfully and saved at {json_path}.")
            st.session_state.patent_name = patent_name
            
        if check_formality:
            st.session_state.check_formality = True
            st.session_state.check_formality_run = False

        if st.session_state.check_formality:
            
            # Create a Patent object with the updated data
            updated_patent = Patent(
                reference_number=patent_name,
                title=patent_title,
                abstract=uploaded_abstract,
                claims=[claim for claim in st.session_state.claims],
                descriptions=[description for description in st.session_state.descriptions],
                image_paths=[image_path for image_path in st.session_state.images],
                pdf_path=st.session_state.patent.pdf_path
            )
            st.session_state.patent_name = patent_name

            with st.spinner("Analyzing patent..."):
                # Run comparison with the uploaded patent object
                
                if st.session_state.check_formality_run == False:
                    
                    examiner = Examiner(updated_patent)

                    # Ensure the directory exists
                    sign_list_dir = os.path.join(DATA_DIR, "sign_lists")
                    os.makedirs(sign_list_dir, exist_ok=True)  # Create the directory if it doesn't exist

                    # Check if the pickle file exists
                    sign_list_path = os.path.join(DATA_DIR, "sign_lists", patent_name + ".pkl")
                    if os.path.exists(sign_list_path):
                        with open(sign_list_path, "rb") as f:
                            sign_list = pickle.load(f)
                    else:
                        my_bar = st.progress(0, text="starting")
                        for status, percent in examiner.oai_extract_signs_from_texts():
                            my_bar.progress(percent, text=status)
                        
                        examiner.oai_extract_signs_from_images()

                        sign_list = examiner.compare_signs_frontend()

                        def remap_to_consecutive_pages(figure_to_page_mapping):
                            length = len(figure_to_page_mapping)
                            pages = sorted(list(set(figure_to_page_mapping)))
                            map_to = list(range(len(pages)))
                            new_mapping = []
                            for i in range(length):
                                for j, page in enumerate(pages):
                                    if figure_to_page_mapping[i] == page:
                                        new_mapping.append(map_to[j])
                            return new_mapping

                        updated_patent.figure_to_page_mapping = remap_to_consecutive_pages(examiner.image_signs.figure_to_page_mapping)
                        json_path = os.path.join(patent_folder, f"{patent_name}.json")
                        with open(json_path, "w") as f:
                            f.write(updated_patent.model_dump_json())
                        sign_list = sorted(sign_list, key=lambda x: (not x.concepts_deviate, x.no_error, str(x.number).zfill(5)))
                        with open(sign_list_path, "wb") as f:
                            pickle.dump(sign_list, f)

                        # Pickle the sign_list
                        sign_list_path = os.path.join(sign_list_dir, patent_name + ".pkl")
                        with open(sign_list_path, "wb") as f:
                            pickle.dump(sign_list, f)

                    st.session_state.check_formality_run = True
                    

                st.success("Analysis completed!")

                # Define the path to the pickle file
                sign_list_path = os.path.join(DATA_DIR, "sign_lists", patent_name + ".pkl")
                st.session_state.patent_name = patent_name
                st.session_state.sign_list_path = sign_list_path
                st.session_state.patent = st.session_state.patent
                st.session_state.evaluation_stage = True
                time.sleep(3)

                js = '''
                <script>
                    var body = window.parent.document.querySelector(".main");
                    console.log(body);
                    body.scrollTop = 0;
                </script>
                '''

                st.components.v1.html(js)
                st.rerun()



if getattr(st.session_state, "evaluation_stage", False):
    path = os.path.join(DATA_DIR, "patents", st.session_state.patent_name + ".json")

        # Check if the JSON file exists
    if os.path.exists(path):
        with open(path) as f:
            st.session_state.patent = Patent.model_validate_json(f.read())
    
    patent = st.session_state.patent
    sign_list_path = st.session_state.sign_list_path
    
    # Check if the file exists before attempting to load
    if os.path.exists(sign_list_path):
        with open(sign_list_path, "rb") as f:
            sign_list = pickle.load(f)
    else:
        print(f"Pickle file not found at {sign_list_path}")

    error_signs = [sign.number for sign in sign_list if sign.no_error==False]
    st.title("PatentPolish - "+st.session_state.patent_name)


    
    col1, col2 = st.columns([5,1])
    with col1:  
        st.warning(f"{len(error_signs) if len(error_signs)>0 else 'No'} sign{'' if len(error_signs)==1 else 's'} with errors: {list_to_string(error_signs)}")

    with col2:
        if st.button("Back to Start"):
            st.session_state.evaluation_stage = False
            st.session_state.check_formality_run = False
            st.session_state.check_formality = False
            st.session_state.sign_list_path = None
            st.session_state.patent = None
            st.rerun()
    

    # Define the columns with a smaller width for column 1
    col1, col2 = st.columns([3, 7])  # Adjust the ratio here: column 1 is 1/4 of the width, column 2 is 3/4
    color_list = [
        "#A8D8EA",  # Light Blue
        "#FADABF",  # Light Orange
        "#F8BBD0",  # Light Pink
        "#E3F2A0",  # Light Green
        "#D7B2E5"   # Light Purple
    ]
    #color_list = ["yellow", "blue", "red", "green", "purple"]


    with col1:


        def get_error_string(sign):
            if sign.no_error:
                return ""
            if sign.sign_only_in_text:
                return ": not in figures"
            if sign.sign_only_in_figure:
                return f": not in text but in figure {sign.figure_occurencies[0].figure}"
            if sign.concepts_deviate:
                return ": concepts deviate"
            return "unknown error"

        option = st.selectbox("Select sign", [str(sign.number)+get_error_string(sign) for sign in sign_list])
        #get sign from sign_list where sign.number == option
        sign = next((sign for sign in sign_list if (option == str(sign.number) or option.startswith(str(sign.number)+": "))), None)

        

        st.write("Found concepts for this sign:")
        for i, key in enumerate(sign.concepts):
            # Cycle through the color_list based on the index
            color = color_list[i % len(color_list)]  # Modulo ensures we cycle through the list if there are more concepts than colors
            
            found_string = f"'{key}'"
            
            # Use st.markdown with inline HTML for colored output
            st.markdown(f"<span style='background-color:{color};'>{found_string}</span>", unsafe_allow_html=True)

        relevant_figures = [occ.figure for occ in sign.figure_occurencies]
        relevant_pages = list(set([patent.figure_to_page_mapping[figure-1]+1 for figure in relevant_figures]))
        if len(relevant_figures) > 0:
            st.info(f"Sign {sign.number} found in figure{"s" if len(sign.figure_occurencies)>1 else ""} {list_to_string(relevant_figures)}")
        else:
            st.info(f"{sign.number} not found in any figures")

        for image_path in patent.image_paths:
            if int(image_path[:-4].split("_")[-1]) in relevant_pages:
                st.image(image_path, caption="Image"+image_path[:-4].split('_')[-1], use_column_width=True)


    with col2:
        # Assign a color to each concept and ensure all occurrences for that concept use the same color
        concept_color_map = {concept: color_list[i % len(color_list)] for i, concept in enumerate(sign.concepts)}

        # Highlight sentences from text_occurencies in abstract, description, or claims
        def highlight_text(text, concept_occurrences, concept_color_map, sign_number):
            """Highlight occurrences based on concept colors and sign number in the given text."""
            
            def streghthen_concept_and_sign_number(concept, sign_number, text):
                variations = [" "+concept+" ", " "+concept+". ", " "+concept+",", " "+concept+"s ", " "+concept+"es ", " "+concept+"s. ", " "+concept+"es. "]
                variations.extend([variation.capitalize() for variation in variations])
                for con in variations:
                    if con in text:
                        text = text.replace(
                            con,
                            f"<strong><u>{con}</u></strong>"
                        )
                text = re.sub(rf'\b{sign_number}\b', f"<strong><u>{sign_number}</u></strong>", text)
                return text

            for concept, occurencies in concept_occurrences.items():
                color = concept_color_map[concept]  # Get the color for the concept
                for occ in occurencies:
                    # Highlight the whole sentence with the concept color
                    sequence_matcher = difflib.SequenceMatcher(None, occ.whole_sentence, text)
                    start_occ, _, size = sequence_matcher.find_longest_match(0, None, 0, None)
                    overlap = occ.whole_sentence[start_occ:start_occ+size]
                    if len(overlap)/len(occ.whole_sentence) > 0.8 and str(sign_number) in overlap:
                        strengthened_overlap = streghthen_concept_and_sign_number(concept, sign_number, overlap)
                        text = text.replace(
                            overlap,
                            f"<span style='background-color:{color}'>{strengthened_overlap}</span>"
                        )
            
            sentences = split_into_sentences(text)
            for concept, occurencies in concept_occurrences.items():
                color = concept_color_map[concept]
                for sentence in sentences:
                    variations = [concept, " "+concept+" ", " "+concept+". ", " "+concept+",", " "+concept+"s ", " "+concept+"es ", " "+concept+"s. ", " "+concept+"es. "]
                    variations.extend([variation.capitalize() for variation in variations])
                    if str(sign_number) in sentence and any([variation in sentence for variation in variations]):
                        strengthened_sentence = streghthen_concept_and_sign_number(concept, sign_number, sentence)
                        text = text.replace(
                            sentence,
                            f"<span style='background-color:{color}'>{strengthened_sentence}</span>"
                        )

                        
            return text

        # Abstract
        abstract_occurencies = {concept: [occ for occ in occ_list if occ.text_type == TextType.ABSTRACT] 
                                for concept, occ_list in sign.concepts.items()}
        is_in_abstract = any([any([occ.text_type == TextType.ABSTRACT for occ in liste]) for liste in sign.concepts.values()])
        highlighted_abstract = highlight_text(
            patent.abstract,
            abstract_occurencies,
            concept_color_map,
            sign.number
        )

        if is_in_abstract:
            st.write("Abstract:")
            abstract_html = f"<div style='background-color:#f0f0f0;color:black;padding:10px;border-radius:5px;'>{highlighted_abstract}</div>"
            st.markdown(abstract_html, unsafe_allow_html=True)

        # Description
        description_occurencies = {concept: [occ for occ in occ_list if occ.text_type == TextType.DESCRIPTION] 
                                for concept, occ_list in sign.concepts.items()}
        descriptions = patent.descriptions.copy()
        highlighted_descriptions = []

        all_desc_list_indices = []
        for concept, occ_list in description_occurencies.items(): 
            all_desc_list_indices.extend([occ.list_index for occ in occ_list if occ.list_index is not None and 0 <= occ.list_index < len(descriptions)])

        disjoint_list_indices = sorted(list(set(all_desc_list_indices)))

        for list_index in disjoint_list_indices:
            highlighted_description = descriptions[list_index]
            for concept, occ_list in description_occurencies.items():
                color = concept_color_map[concept]  # Use the same color for the concept
                highlighted_description = highlight_text(   
                    highlighted_description,
                    description_occurencies,
                    concept_color_map,
                    sign.number
                )
            highlighted_descriptions.append(highlighted_description)

        if len(highlighted_descriptions) > 0:   
            st.write("Description:")
            description_html = f"<div style='background-color:#f0f0f0;color:black;padding:10px;border-radius:5px;'>{'<br>'.join(highlighted_descriptions)}</div>"
            st.markdown(description_html, unsafe_allow_html=True)

        # Claims
        claims_occurencies = {concept: [occ for occ in occ_list if occ.text_type == TextType.CLAIMS] 
                            for concept, occ_list in sign.concepts.items()}
        claims = patent.claims.copy()
        highlighted_claims = []

        all_claim_list_indices = [] 
        for concept, occ_list in claims_occurencies.items(): 
            all_claim_list_indices.extend([occ.list_index for occ in occ_list if occ.list_index is not None and 0 <= occ.list_index < len(claims)])

        disjoint_claim_list_indices = sorted(list(set(all_claim_list_indices)))

        for list_index in disjoint_claim_list_indices:
            highlighted_claim = claims[list_index]
            for concept, occ_list in claims_occurencies.items():
                color = concept_color_map[concept]  # Use the same color for the concept
                highlighted_claim = highlight_text(
                    highlighted_claim,
                    claims_occurencies,
                    concept_color_map,
                    sign.number
                )
            highlighted_claims.append(highlighted_claim)

        if len(highlighted_claims) > 0:
            st.write("Claims:")
            claims_html = f"<div style='background-color:#f0f0f0;color:black;padding:10px;border-radius:5px;'>{'<br>'.join(highlighted_claims)}</div>"
            st.markdown(claims_html, unsafe_allow_html=True)

    # Path where the CSV will be stored
    patent_name = st.session_state.patent_name
    csv_file_path = os.path.join(DATA_DIR, "sign_lists", f"{patent_name}_sign_analysis.csv")

    # Generate the CSV file
    generate_sign_csv(sign_list, csv_file_path)

    # Read the generated CSV file into a pandas dataframe
    df = pd.read_csv(csv_file_path)

    # Apply the custom conversion function to the "Reference sign" column
    df['Reference sign'] = df['Reference sign'].apply(convert_to_int_if_possible)

    # Sort the dataframe by the "Reference sign" column, using mixed type sorting (numeric first, then strings)
    df_sorted = df.sort_values(by="Reference sign", key=lambda col: col.map(convert_to_int_if_possible))

    # Use Streamlit's markdown to expand the dataframe width using custom CSS
    st.markdown(
        """
        <style>
        .dataframe-container {
            width: 100%;
            max-width: 100%;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.subheader("List of reference signs with occurencies")
    # Display the dataframe with full width
    st.dataframe(df_sorted, hide_index=True, width=1800)
    
        
                


