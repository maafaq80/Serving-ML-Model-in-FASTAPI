from fastapi import FastAPI
from typing import Annotated,Literal
from pydantic import Field,BaseModel,computed_field
import pandas as pd
import pickle
from fastapi.responses import JSONResponse

with open('model.pkl',"rb") as f:
    model=pickle.load(f)
    
tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

app=FastAPI()

## Creating pydantic model
class User_Input(BaseModel):
    
    age: Annotated[int,Field(...,gt=0,lt=120,description="Age of the user")]
    weight:Annotated[float,Field(...,gt=0,description="weight of the user")]
    height:Annotated[float,Field(...,gt=0,description="height of the user")]
    income_lpa:Annotated[float,Field(...,gt=0,description="income per annum of the user")]
    smoker:Annotated[bool,Field(...,description="Is User smoker or not ")]
    city:Annotated[str,Field(...,description="city of residence of the user ")]
    occupation:Annotated[Literal['retired', 'freelancer', 'student', 'government_job','business_owner', 'unemployed', 'private_job'],Field(...,description="Current Employment status")]
    
    
    @computed_field
    @property
    def bmi(self)->float:
        return self.weight/(self.height**2)
    
    @computed_field
    @property
    def lifestyle_risk(self)->str:
        if(self.smoker and self.bmi > 30):
            return "high"
        elif(self.smoker or self.bmi > 27):
            return "medium"
        else:
            return "low"
        
    @computed_field
    @property
    def age_group(self)->str:
            if self.age < 25:
                return "young"
            elif self.age < 45:
                return "adult"
            elif self.age < 60:
                return "middle_aged"
            return "senior"
        
    @computed_field
    @property  
    def city_tier(self)->str:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3
        


@app.post("/predict")
def predict_premium(data: User_Input):
    input_data = pd.DataFrame({
        'bmi': [data.bmi],
        'age_group': [data.age_group],
        'lifestyle_risk': [data.lifestyle_risk],
        'city_tier': [data.city_tier],
        'income_lpa': [data.income_lpa],
        'occupation': [data.occupation]
    })

    prediction = model.predict(input_data)
    return JSONResponse(status_code=200, content={"predicted_category": prediction[0]})
