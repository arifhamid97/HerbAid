
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import torch
from transformers import pipeline
from PIL import Image
from transformers import ImageClassificationPipeline
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()
class MyPipeline(ImageClassificationPipeline):
    """

    Overwriting the hf pipeline.
    By default hf model always return logits.
    To make it simple we normalize the logit at the end of the pipeline
    
    """
    def postprocess(self, model_outputs):
        predicted_label = model_outputs[0].argmax(-1).item()
        predicted_class = self.model.config.id2label[predicted_label]
        return predicted_class

class HerbDetails(BaseModel):
    plant_name: str = Field(description="The plant Name without underscore")
    usage: str = Field(description="Answer to the herb plant usage")
    how_to_use: str = Field(description="Answer to the how to use the herb plant")


class HerbAid:
    def __init__(self) -> None:
        self.hf_model = os.getenv("HF_MODEL")
        self.open_ai_key = os.getenv("OPENAI_API_KEY")
        self.open_model = os.getenv("OPENAI_MODEL")

    def __init_open_ai(self):
        model = ChatOpenAI(openai_api_key=self.open_ai_key, model=self.open_model)
        return model

    def __analyse_image(self, image:Image) -> str:
        vision_classifier = pipeline(
            task="image-classification",
            model=self.hf_model,
            feature_extractor=self.hf_model,
            device=torch.device("mps"),
            pipeline_class=MyPipeline,
            model_kwargs={"return_dict":False}
            )
        
        return vision_classifier(image)

    def invoke_chain(self,image):

        plant_classification = self.__analyse_image(image)

        print(plant_classification)

        if plant_classification =="Not_a_plant":
            return None

        parser = JsonOutputParser(pydantic_object=HerbDetails)
        template = """
        What is {plant_name} usage and step to use when you lost in rainforest and during emergency situation.
        {format_instructions}

        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["plant_name"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        model = self.__init_open_ai()

        chain = prompt | model | parser
        return chain.invoke({"plant_name":plant_classification})
