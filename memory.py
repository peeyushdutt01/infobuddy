from openai import OpenAI

long_term = {
    "user_prefrences": [],
    "user_goals": [],
    "custom_system_prompt" : "",
    "favourite_topics": [],
    "user_personal_details": {
        "user_name": "",
        "user_nickname": "",
        "user_current_designation": "",
        "user_email_adress": [],
        "user_phone_no":[],
        "user_location": "",
        "user_instagram_url":"",
        "user_linkedin_url":"",
        "user_long_term_goal": "",
    },
    "custom_persona": "",
    "user_summary": ""
}


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # can be any string
)

def ask_llm(conversation: list[dict],long_term, model: str = "qwen3:latest"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role" : "system", "content":"""you are a memory management agent , and our goal is to maintain the memory state for the llm,
             the long term memory is incremental , try to update the feilds with the new information , while preserving the previous information , make sure the important details are not ignored or lost, check the formatting before creating the output , and only return the output in the required json format , no extra text or explaination is needed , just the right formatted json, no follow ups or reasking , just take the query , extract if something useful and new is found , and then return the updated json. 
             you have to store the user's long term memory in a structured format JSON, the structure is already defined, it is like :
             {
                "user_prefrences": list[str],
                "user_goals": list[str],
                "custom_system_prompt" : str,
                "favourite_topics": list[str],
                "user_personal_details": {
                    "user_name": str,
                    "user_nickname": str,
                    "user_current_designation": str,
                    "user_email_adress": list[str],
                    "user_phone_no":list[str],
                    "user_location": str,
                    "user_instagram_url": str,
                    "user_linkedin_url": str,
                    "user_long_term_goal": str,
                },
                "custom_persona": str,
                "user_summary": str
            }"""},
            {"role": "user", "content": f"here is the conversation chunk : {conversation} , and here is the existing memory : {long_term}"}
        ]
    )

    return response.choices[0].message.content

conversation = [
  {
    "role": "user",
    "content": "Hey, I'm Rahul Sharma. I recently moved to Bangalore from Pune for a new software engineering job."
  },
  {
    "role": "ai",
    "content": "Congratulations on the new job and the move to Bangalore! How are you settling in?"
  },
  {
    "role": "user",
    "content": "Pretty well. I'm 28 years old and working as a backend engineer at a fintech startup. I mostly code in Go and Python."
  },
  {
    "role": "ai",
    "content": "That sounds exciting. Go and Python are a strong combination for backend systems."
  },
  {
    "role": "user",
    "content": "Outside work, I love hiking, photography, and playing badminton on weekends."
  },
  {
    "role": "ai",
    "content": "Those are great hobbies. Bangalore also has a lot of nearby trekking spots."
  },
  {
    "role": "user",
    "content": "Yeah, and I'm trying to visit at least one new trekking destination every month."
  },
  {
    "role": "ai",
    "content": "That's a nice goal. Have you explored any favorites so far?"
  },
  {
    "role": "user",
    "content": "Kodachadri has been my favorite so far. Also, I'm vegetarian and enjoy trying different South Indian restaurants."
  },
  {
    "role": "ai",
    "content": "Kodachadri is beautiful. Bangalore should give you plenty of vegetarian food options too."
  },
  {
    "role": "user",
    "content": "I live with my golden retriever, Bruno. He's 3 years old and keeps me active."
  },
  {
    "role": "ai",
    "content": "Bruno sounds like a wonderful companion."
  },
  {
    "role": "user",
    "content": "I'm also planning to prepare for technical interviews because I want to switch to a larger company within the next year."
  },
  {
    "role": "ai",
    "content": "That's a valuable long-term objective. Consistent preparation can make a big difference."
  },
  {
    "role": "user",
    "content": "By the way, today I'm feeling exhausted because I stayed up until 2 AM finishing a project deadline."
  },
  {
    "role": "ai",
    "content": "That sounds like a long night. Hopefully you can get some rest soon."
  },
  {
    "role": "user",
    "content": "My favorite music genre is indie rock, and my favorite band is The Local Train."
  },
  {
    "role": "ai",
    "content": "Nice choice. Indie rock has a lot of great artists to explore."
  }
]

long_term = ask_llm(conversation,long_term)


print(long_term)