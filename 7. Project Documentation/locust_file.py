from locust import HttpUser, task, between

class CreditCardUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_approval(self):
        # Form data payload
        payload = {
            "applicant_name": "TestUser",
            "income_type": "Working",
            "education_type": "Higher education",
            "annual_income": "60000"
        }
        self.client.post("/predict", data=payload)
