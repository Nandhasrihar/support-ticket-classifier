import pandas as pd
import random

random.seed(42)

templates = {
    "Billing": {
        "subjects": [
            "Payment Failed","Refund Request","Invoice Issue",
            "Subscription Problem","Double Charge","Billing Error",
            "Payment Pending","Incorrect Invoice"
        ],
        "verbs": [
            "charged twice","billed incorrectly","payment failed",
            "refund delayed","invoice missing","subscription renewed unexpectedly"
        ],
        "objects": [
            "monthly subscription","annual plan","premium account",
            "invoice","credit card payment","GST invoice"
        ]
    },

    "Technical": {
        "subjects": [
            "Login Issue","App Crash","Server Error",
            "Password Reset","Website Bug","Upload Failed",
            "Dashboard Error","Performance Issue"
        ],
        "verbs":[
            "cannot login","app crashes","website freezes",
            "password reset not working","upload fails",
            "dashboard shows error","server unavailable"
        ],
        "objects":[
            "after update","while uploading files",
            "during checkout","on mobile app",
            "after entering password","on Chrome browser"
        ]
    },

    "HR":{
        "subjects":[
            "Leave Request","Salary Slip","Attendance",
            "Holiday List","Offer Letter","Employee ID",
            "Resignation","Work From Home"
        ],
        "verbs":[
            "need","request","unable to access",
            "missing","waiting for","want to apply for"
        ],
        "objects":[
            "annual leave","salary slip","holiday calendar",
            "employee ID","offer letter",
            "attendance correction","WFH approval"
        ]
    },

    "General":{
        "subjects":[
            "Feedback","General Question","Help",
            "Contact Details","Working Hours",
            "Suggestion","Documentation","Thank You"
        ],
        "verbs":[
            "need","want","looking for",
            "thank you for","appreciate","can you provide"
        ],
        "objects":[
            "office timings","documentation",
            "contact information","general information",
            "customer support","help regarding services"
        ]
    }
}

rows=[]

for category,data in templates.items():

    for i in range(125):

        subject=random.choice(data["subjects"])

        body=f"I {random.choice(data['verbs'])} regarding my {random.choice(data['objects'])}."

        rows.append([subject,body,category])

random.shuffle(rows)

df=pd.DataFrame(rows,columns=["subject","body","category"])

df.to_csv("data/tickets.csv",index=False)

print(df.head())
print("Dataset Created:",len(df))