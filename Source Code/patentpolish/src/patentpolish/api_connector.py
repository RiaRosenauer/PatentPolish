import os
from xml.etree import ElementTree

import fitz
from oauthlib.oauth2 import BackendApplicationClient
from pypdf import PdfReader, PdfWriter
from requests_oauthlib import OAuth2Session

from patentpolish.patent_model import Patent
from patentpolish.util import logger

AUTH_URL = "https://ops.epo.org/3.2/auth/accesstoken"
BASE_PATH = "https://ops.epo.org/3.2/rest-services"


def pdf_cat(input_files, output_stream):
    input_streams = []
    try:
        for input_file in input_files:
            input_streams.append(open(input_file, 'rb'))
        writer = PdfWriter()
        for reader in map(PdfReader, input_streams):
            for n in range(len(reader.pages)):
                writer.add_page(reader.pages[n])
        writer.write(output_stream)
    finally:
        for f in input_streams:
            f.close()
        output_stream.close()


class ApiConnector:

    def __init__(self) -> None:
        client = BackendApplicationClient(client_id=os.getenv('EPO_CONSUMER_KEY'))
        oauth = OAuth2Session(client=client)
        oauth.fetch_token(token_url=AUTH_URL, client_id=os.getenv('EPO_CONSUMER_KEY'),
                          client_secret=os.getenv('EPO_CONSUMER_SECRET_KEY'))
        self.oauth = oauth


    def get_patent(self, reference_number) -> Patent:
        # check if patent lies in folder data/patents as json
        path = os.path.join("data", "patents", reference_number+".json")
        if os.path.exists(path):
            with open(path) as f:
                patent = Patent.model_validate_json(f.read())
            
            if os.path.exists(patent.pdf_path) and all([os.path.exists(image_path) for image_path in patent.image_paths]):
                logger.info(f"Patent {reference_number} loaded from file")
                return patent
        
        patent = self.request_patent(reference_number)
        os.makedirs(os.path.join("data", "patents"), exist_ok=True)
        with open(path, "w") as f:
            f.write(patent.model_dump_json())
        return patent

    
    def request_patent(self, reference_number: str):
        logger.critical(f"Retrieving patent {reference_number} from EPO API")
        return Patent(reference_number = reference_number,
                    title = self.request_title(reference_number),
                    abstract = self.request_abstract(reference_number),
                    claims = self.request_claims(reference_number),
                    descriptions = self.request_description(reference_number),
                    image_paths = self.request_images(reference_number),
                    pdf_path = self.request_pdf(reference_number)
                    )


    def request_title(self, reference_number: str) -> str:
        r = self.oauth.get(BASE_PATH + "/published-data/publication/epodoc/" + reference_number + "/biblio")
        tree = ElementTree.ElementTree(ElementTree.fromstring(r.content))
        for child in tree.iter():
            if child.tag == "{http://www.epo.org/exchange}invention-title" and child.attrib.get("lang") == "en":
                return child.text
        for child in tree.iter():
            logger.info(child.tag)
        logger.error(f"No title found for patent {reference_number}")
        return None
    
    def request_abstract(self, reference_number: str) -> str:
        r = self.oauth.get(BASE_PATH + "/published-data/publication/epodoc/" + reference_number + "/abstract")
        tree = ElementTree.ElementTree(ElementTree.fromstring(r.content))
        for child in tree.iter():
            if child.tag == "{http://www.epo.org/exchange}abstract":
                for abstract in child:
                    if abstract.tag == "{http://www.epo.org/exchange}p":
                        return abstract.text
        for child in tree.iter():
            logger.info(child.tag)
        logger.error(f"No abstract found for patent {reference_number}")
        return None
    
    def request_claims(self, reference_number: str) -> str:
        r = self.oauth.get(BASE_PATH + "/published-data/publication/epodoc/" + reference_number + "/claims")
        tree = ElementTree.ElementTree(ElementTree.fromstring(r.content))
        for child in tree.iter():
            if child.tag == "{http://www.epo.org/fulltext}claims":
                for claim in child:
                    if claim.tag == "{http://www.epo.org/fulltext}claim":
                        claims = []
                        for claim_text in claim:
                            if claim_text.tag == "{http://www.epo.org/fulltext}claim-text":
                                claims.append(claim_text.text)
                        return claims
        for child in tree.iter():
            logger.info(child.tag)
        logger.error(f"No claims found for patent {reference_number}")
        return None
    
    def request_description(self, reference_number: str) -> str:
        r = self.oauth.get(BASE_PATH + "/published-data/publication/epodoc/" + reference_number + "/description")
        tree = ElementTree.ElementTree(ElementTree.fromstring(r.content))
        for child in tree.iter():
            if child.tag == "{http://www.epo.org/fulltext}description":
                descriptions = []
                for description in child:
                    if description.tag == "{http://www.epo.org/fulltext}p":
                        descriptions.append(description.text)
                return descriptions
        
        for child in tree.iter():
            logger.info(child.tag)
        logger.error(f"No description found for patent {reference_number}")
        return None
    
    def request_images(self, reference_number: str):
        r = self.oauth.get(BASE_PATH + "/published-data/publication/epodoc/" + reference_number + "/images")
        tree = ElementTree.ElementTree(ElementTree.fromstring(r.content))
        ElementTree.indent(tree, space="\t", level=0)

        for child in tree.iter():
            if child.tag == "{http://ops.epo.org}document-instance" and child.attrib.get("desc") == "Drawing":
                pages = child.attrib.get("number-of-pages")
                png_paths = []
                for i in range(1, int(pages)+1):
                    url = BASE_PATH+"/"+child.attrib.get("link")+"?Range="+str(i)
                    img_r= self.oauth.get(url)
                    pdf_path = os.path.join("data", "images", reference_number+"_"+str(i)+".pdf")
                    os.makedirs(os.path.join("data", "images"), exist_ok=True)
                    with open(pdf_path, "wb") as f:
                        f.write(img_r.content)
                    with fitz.open(pdf_path) as doc:
                        page = doc.load_page(0)  # number of page
                        pix = page.get_pixmap(matrix=fitz.Matrix(150/72,150/72))
                        output = os.path.join("data", "images", reference_number+"_"+str(i)+".png")
                        pix.save(output)
                        png_paths.append(output)
                    os.remove(pdf_path)
                return png_paths
                
        for child in tree.iter():
            logger.info(child.tag)
        logger.error(f"No images found for patent {reference_number}")
        return None
    
    def request_pdf(self, reference_number: str):
        r = self.oauth.get(BASE_PATH + "/published-data/publication/epodoc/" + reference_number + "/images")
        tree = ElementTree.ElementTree(ElementTree.fromstring(r.content))
        ElementTree.indent(tree, space="\t", level=0)

        #TODO: Make sure to get all pages of the pdf
        for child in tree.iter():
            if child.tag == "{http://ops.epo.org}document-instance" and child.attrib.get("desc") == "FullDocument":
                pages = child.attrib.get("number-of-pages")
                save_paths = []
                for i in range(1, int(pages)+1):
                    url = BASE_PATH+"/"+child.attrib.get("link")+"?Range="+str(i)
                    img_r= self.oauth.get(url)
                    save_path = os.path.join("data", "pdfs", reference_number+"_"+str(i)+".pdf")
                    os.makedirs(os.path.join("data", "pdfs"), exist_ok=True)
                    with open(save_path, "wb") as f:
                        f.write(img_r.content)
                    save_paths.append(save_path)

                concat_save_path = os.path.join("data", "pdfs", reference_number+".pdf")
                with open(concat_save_path, 'wb') as output_stream:
                    pdf_cat(save_paths, output_stream)

                for pdf in save_paths:
                    os.remove(pdf)

                return concat_save_path
                
        for child in tree.iter():
            logger.info(child.tag)
        logger.error(f"No fullimage found for patent {reference_number}")
        return None



if __name__ == "__main__":
    api_connector = ApiConnector()
    id = "EP4353111"

    patent = api_connector.get_patent(id)
    print("TITLE:\n", patent.title.upper(), "\n")
    print("ABSTRACT:\n", patent.abstract, "\n")
    print("CLAIMS:\n", patent.claims, "\n")
    print("DESCRIPTIONS:\n", patent.descriptions, "\n")
