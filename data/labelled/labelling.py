import json
import openai

API_KEY = ''


def main():
    openai.api_key = API_KEY

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": 
                    "You are an assistant who helps me label comments of hotels in the following 8 aspects: \n" + 
                    "- Cleanliness\n"+
                    "- Food\n"+
                    "- Service and Staff\n"+
                    "- Amenities\n"+
                    "- Location\n"+
                    "- Room Comfort\n"+
                    "- Parking\n"+
                    "- Security\n"+
                    "You will rate each aspect from 0 to 1, with 0 being terrible and 1 being amazing. If the aspect is no mentioned, use 0.5.\n"+
                    "I will give you a comma-delimited list of comments and I want you to rate each comment and respond in the json format with a list of json object.\n"+
                    "Each json object should have a field called \"comment\" which contains the full comment and a field calld \"labels\" which should be a json object containing the rating of all 8 aspects"
            },
            {
                "role": "user",
                "content": 
                    "\"Very comfy bed large shower\","+
                    "\"Our room is clean and the bed is so comfortable\""
            },
        ]
    )

    print(completion.choices[0].message.content)


if __name__ == '__main__':
    with open('./config.json', mode='r') as config_file:
        config = json.load(config_file)
        API_KEY = config['API_KEY']
    main()