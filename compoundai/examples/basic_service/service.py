from compoundai.sdk.service import service
from compoundai.sdk.decorators import nova_endpoint
from compoundai.sdk.dependency import depends
from bentoml import api

@service(
    nova={
        "enabled": True,
        "namespace": "example",
    }
)
class ModelService:
    def __init__(self):
        print("Model service init")

    @nova_endpoint()
    def predict(self, text: str) -> str:
        """Nova endpoint for prediction"""
        return f"Nova processed: {text}"

    @nova_endpoint(name="custom")
    def custom_endpoint(self, data: dict) -> dict:
        """Custom Nova endpoint"""
        return {"result": data}
    
    @api
    def bentoml_predict(self, text: str) -> str:
        """Regular BentoML API endpoint"""
        return f"BentoML processed: {text}"
    
    @api(name="status")
    def get_status(self) -> dict:
        """Regular BentoML API with custom name"""
        return {"status": "healthy"}

@service()
class PipelineService:
    # Depend on the ModelService
    model = depends(ModelService)
    
    def __init__(self):
        print("Pipeline service init")

    @nova_endpoint()
    def process(self, text: str) -> dict:
        """Process text using Nova endpoints"""
        # Call Nova endpoints
        nova_result = self.model.predict(text)
        nova_extra = self.model.custom_endpoint({"input": text})
        
        return {
            "nova_prediction": nova_result,
            "nova_extra": nova_extra
        }
    
    @api
    def process_bentoml(self, text: str) -> dict:
        """Process text using regular BentoML APIs"""
        # Call regular BentoML APIs
        bentoml_result = self.model.predict(text)
        # status = self.model.get_status()
        
        return {
            "bentoml_prediction": bentoml_result,
            # "service_status": status
        } 