import os
import json
from openai import OpenAI
import csv
import io
import zipfile
import httpx

base_url = "https://api.ai.it.ufl.edu/v1"
RASHI_LITELLM_KEY = os.getenv('RASHI_LITELLM_KEY')
client = OpenAI(
    api_key=RASHI_LITELLM_KEY,
    base_url=base_url
)

knowledge_data = {
    "Knowledge_1": {
        "description": "“Informed consent” means that I am given information about the trial so I can freely decide whether to participate.",
        "correctValue": 1,
    },
    "Knowledge_2": {
        "description": "“Standard treatments” are the best treatments currently known for a cancer.",
        "correctValue": 1,
    },
    "Knowledge_3": {
        "description": "Standard treatments are never as good as new research treatments.",
        "correctValue": 2,
    },
    "Knowledge_4": {
        "description": "Treatments used in clinical trials may cause side effects.",
        "correctValue": 1,
    },
    "Knowledge_5": {
        "description": "It is up to me to decide whether to be in a clinical trial.",
        "correctValue": 1,
    },
    "Knowledge_6": {
        "description": "Patients in clinical trials must get their care at different places from patients getting standard treatments.",
        "correctValue": 2,
    },
    "Knowledge_7": {
        "description": "If I were to join a clinical trial, I could decide to stop at any time.",
        "correctValue": 1,
    },
    "Knowledge_8": {
        "description": "“Randomization” means that my treatment will be chosen by chance.",
        "correctValue": 1,
    },
    "Knowledge_9": {
        "description": "Once I join a clinical trial, my own doctor will not know what happens to me.",
        "correctValue": 2,
    },
    "Knowledge_10": {
        "description": "Most clinical trials involve a placebo (sugar pill).",
        "correctValue": 2,
    },
    "Knowledge_11": {
        "description": "Side effects in clinical trials are usually worse than with standard treatments.",
        "correctValue": 2,
    },
    "Knowledge_12": {
        "description": "Clinical trials are only used as a last resort.",
        "correctValue": 2,
    },
    "Knowledge_13": {
        "description": "The only way to find out about clinical trials is from my doctor.",
        "correctValue": 2,
    },
    "Knowledge_14": {
        "description": "Clinical trials are not appropriate for patients with cancer.",
        "correctValue": 2,
    },
    "Knowledge_15": {
        "description": "My doctor can start a clinical trial without the approval of professionals who protect patient rights.",
        "correctValue": 2,
    },
    "Knowledge_16": {
        "description": "A clinical trial is available for anyone with cancer who wants to take part.",
        "correctValue": 2,
    },
    "Knowledge_17": {
        "description": "Institutional Review Boards review and monitor clinical trials to keep patients safe.",
        "correctValue": 1,
    },
    "Knowledge_18": {
        "description": "Informed Consent mainly protects researchers from lawsuits.",
        "correctValue": 2,
    },
    "Knowledge_19": {
        "description": "Clinical trials are done to improve standard treatments.",
        "correctValue": 1,
    },
}

attitudes_data = {
    "Attitudes_77": "I believe clinical trials are only for people with cancer that cannot be treated any other way.",
    "Attitudes_78": "I do not trust the medical system.",
    "Attitudes_79": "I am worried that the treatment I receive in a clinical trial would not work for me.",
    "Attitudes_80": "I am concerned that participating in a clinical trial may be dangerous.",
    "Attitudes_81": "I have concerns about the time, transportation, and/or travel required to participate in a clinical trial.",
    "Attitudes_82": "I am worried about the financial burden of participating in a clinical trial.",
    "Attitudes_83": "I'm worried that my family wouldn't want me to go on a clinical trial.",
    "Attitudes_84": "I am concerned about the privacy of my personal medical information if I participate in a clinical trial.",
    "Attitudes_85": "I'm worried that my medical care won't be as good if I join a clinical trial.",
    "Attitudes_86": "I am concerned about being treated as an experiment rather than a person in a clinical trial.",
    "Attitudes_87": "I'm afraid I'll get a sugar pill (placebo) instead of real medicine on a clinical trial.",
    "Attitudes_88": "I wouldn't ask about clinical trials unless my doctor brought them up first.",
    "Attitudes_89": "I don't like to try new treatments until they've been around for a while.",
    "Attitudes_90": "I am not familiar with what clinical trials are.",
    "Attitudes_91": "I'm afraid that if I take part in a clinical trial my treatment will be selected at random by a computer rather than by my doctor.",
}

def normalize_goal_response(data: dict):
    goals = data.get("suggestedGoals", [])

    normalized = []
    for i, goal in enumerate(goals):
        normalized.append({
            "id": (goal.get("id") or goal.get("goalId") or f"goal-{i}").strip(),
            "title": (goal.get("title") or goal.get("goalTitle") or "Goal").strip(),
            "description": (
                goal.get("description")
                or goal.get("goalDescription")
                or ""
            ).strip(),
            "confidence": goal.get("confidence"),
            "reason": (goal.get("reason") or "").strip(),
        })

    return {"suggestedGoals": normalized}

async def export_survey_from_qualtrics():
    api_token = os.getenv("QUALTRICS_APIKEY")
    data_center = os.getenv("QUALTRICS_DATACENTER")
    survey_id = os.getenv("QUALTRICS_PRESURVEY_ID", "SV_881ymxhR2uCez30")

    if not api_token or not data_center:
        raise ValueError("Missing Qualtrics API key or datacenter")

    file_format = "csv"

    base_url = (
        f"https://{data_center}.qualtrics.com/API/v3/"
        f"surveys/{survey_id}/export-responses/"
    )

    headers = {
        "content-type": "application/json",
        "x-api-token": api_token,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        export_response = await client.post(
            base_url,
            json={"format": file_format},
            headers=headers,
        )
        export_response.raise_for_status()

        progress_id = export_response.json()["result"]["progressId"]

        progress_status = "inProgress"
        file_id = None

        while progress_status not in ["complete", "failed"]:
            progress_response = await client.get(
                base_url + progress_id,
                headers=headers,
            )
            progress_response.raise_for_status()

            progress_result = progress_response.json()["result"]
            progress_status = progress_result["status"]

            print("Download is", progress_result.get("percentComplete"), "complete")

            if progress_status == "complete":
                file_id = progress_result["fileId"]

        if progress_status == "failed":
            raise RuntimeError("Qualtrics export failed")

        download_response = await client.get(
            base_url + file_id + "/file",
            headers=headers,
        )
        download_response.raise_for_status()

    zip_bytes = io.BytesIO(download_response.content)

    with zipfile.ZipFile(zip_bytes) as zip_file:
        csv_filename = zip_file.namelist()[0]

        with zip_file.open(csv_filename) as csv_file:
            decoded_file = io.TextIOWrapper(csv_file, encoding="utf-8-sig")
            rows = list(csv.DictReader(decoded_file))

    print(f"Downloaded {len(rows)} rows.")
    if rows:
        print("Columns:", list(rows[0].keys()))

    return rows

async def get_presurvey_row_from_qualtrics(response_id: str):
    rows = await export_survey_from_qualtrics()

    for index, row in enumerate(rows):
        # Qualtrics CSVs often have 2 metadata rows after the header.
        # This matches your old JS logic that skipped rowIndex 1 and 2.
        if index in [0, 1]:
            continue

        if row.get("ResponseId") != response_id:
            continue

        filtered_row = {}

        for key, value in row.items():
            if (
                key == "ResponseId"
                or key.startswith("Knowledge_")
                or key.startswith("Attitudes_")
            ):
                filtered_row[key] = value

        return filtered_row

    raise ValueError(f"No matching ResponseId found: {response_id}")

def score_presurvey_row(row: dict):
    correct = []
    wrong = []
    unsure = []
    attitudes = []

    for question_id, question_info in knowledge_data.items():
        if question_id not in row:
            continue

        raw_response = row.get(question_id)

        if raw_response in [None, ""]:
            continue

        try:
            response = int(raw_response)
        except ValueError:
            continue

        result = {
            "question": question_id,
            "description": question_info["description"],
            "correctValue": question_info["correctValue"],
            "userResponse": response,
        }

        if response == question_info["correctValue"]:
            correct.append(result)
        elif response == 3:
            unsure.append(result)
        else:
            wrong.append(result)

    for attitude_id, description in attitudes_data.items():
        if attitude_id not in row:
            continue

        raw_score = row.get(attitude_id)

        if raw_score in [None, ""]:
            continue

        try:
            score = int(raw_score)
        except ValueError:
            continue

        if score > 50:
            attitudes.append(
                {
                    "question": attitude_id,
                    "description": description,
                    "score": score,
                }
            )

    return {
        "correct": correct,
        "wrong": wrong,
        "unsure": unsure,
        "attitudes": attitudes,
    }

async def generate_goals_from_scores(scores: dict):
    prompt = f"""
        You are helping personalize the initial learning goals for a clinical trials educational conversation.

        Your job is to recommend the 6 most relevant learning goals based ONLY on the participant's pre-survey responses.

        These goals will appear before the conversation begins and should represent broad topics the participant may want to understand, not specific questions they should ask.

        Imagine each title is the heading of the participant's learning plan. Each goal should be broad enough that multiple conversation questions could fit underneath it.

        Consider:
        - Incorrect knowledge responses.
        - "Unsure" knowledge responses.
        - Strong concerns reflected in attitude items.

        Favor goals that address misconceptions, uncertainty, or high-priority concerns.

        Pre-survey results:

        Wrong knowledge:
        {json.dumps(scores["wrong"], indent=2)}

        Unsure knowledge:
        {json.dumps(scores["unsure"], indent=2)}

        High concern attitudes:
        {json.dumps(scores["attitudes"], indent=2)}

        Return ONLY valid JSON in exactly this format:

        {{
        "suggestedGoals": [
            {{
            "id": "doctor-role",
            "title": "Understand my doctor's role in trials",
            "description": "Learn how your regular doctor and the research team work together throughout a clinical trial.",
            "confidence": 0.96,
            "reason": "The participant expressed uncertainty about who manages care during a clinical trial."
            }}
        ]
        }}

        Rules:
        - Return exactly 6 goals whenever possible.
        - If fewer than 6 goals are strongly supported by the participant's responses, return only the goals that are well justified.
        - Sort goals from highest confidence to lowest confidence.
        - id must be unique within this response.
        - id should be a short lowercase slug based on the title, using only letters, numbers, and hyphens.
        - id should not include spaces.
        - title should be 3–8 words, participant-friendly, and describe a broad learning goal.
        - Titles should represent topics to understand, not individual questions.
        - Prefer titles beginning with "Understand..." or "Learn..." when natural.
        - Good examples:
        - Understand my doctor's role in trials
        - Learn how clinical trial costs work
        - Understand informed consent
        - Learn where clinical trial care happens
        - Understand my rights as a participant
        - Avoid overly specific titles that focus on a single question, such as:
        - Learn how my doctor stays involved
        - Find out whether I'll get a placebo
        - Know when I can leave a trial
        - description should be one concise sentence explaining what the participant would learn or discuss under that goal.
        - The description should expand on the goal, not simply restate the title.
        - confidence should reflect how strongly the participant's pre-survey responses support recommending this goal relative to the other goals.
        - reason should explicitly reference the participant's incorrect, unsure, or high-concern responses.
        - Do not include any text outside the JSON.
        """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You return only valid JSON. No markdown.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    return normalize_goal_response(json.loads(content))

async def generate_more_goals_from_scores(scores: dict, existing_goals: list):
    existing_goal_ids = [goal.get("id") for goal in existing_goals]
    existing_goal_titles = [goal.get("title") for goal in existing_goals]

    prompt = f"""
        You are helping suggest additional learning goals for a clinical trials educational conversation.

        The participant already has these suggested goals:

        Existing goal IDs:
        {json.dumps(existing_goal_ids, indent=2)}

        Existing goal titles:
        {json.dumps(existing_goal_titles, indent=2)}

        Based ONLY on the participant's pre-survey responses, recommend up to 3 additional learning goals.

        These goals will appear before the conversation begins and should represent broad topics the participant may want to understand, not specific questions they should ask.

        Imagine each title is the heading of the participant's learning plan. Each goal should be broad enough that multiple conversation questions could fit underneath it.

        Do not repeat existing goal IDs, existing titles, or very similar goal topics.

        Pre-survey results:

        Wrong knowledge:
        {json.dumps(scores["wrong"], indent=2)}

        Unsure knowledge:
        {json.dumps(scores["unsure"], indent=2)}

        High concern attitudes:
        {json.dumps(scores["attitudes"], indent=2)}

        Return ONLY valid JSON in exactly this format:

        {{
        "suggestedGoals": [
            {{
            "id": "costs-and-visits",
            "title": "Understand clinical trial costs",
            "description": "Learn what costs, travel, visits, and time commitments may come with joining a clinical trial.",
            "confidence": 0.88,
            "reason": "The participant reported concern about financial burden or travel."
            }}
        ]
        }}

        Rules:
        - Return up to 3 goals.
        - Do not repeat any existing id.
        - Do not repeat any existing title.
        - Do not recommend goals that substantially overlap with existing goal topics.
        - If no strong additional goals are supported, return an empty suggestedGoals list.
        - Sort from highest confidence to lowest confidence.
        - id must be unique within this response.
        - id should be a short lowercase slug based on the title, using only letters, numbers, and hyphens.
        - id should not include spaces.
        - title should be 3–8 words, participant-friendly, and describe a broad learning goal.
        - Titles should represent topics to understand, not individual questions.
        - Prefer titles beginning with "Understand..." or "Learn..." when natural.
        - Good examples:
        - Understand my doctor's role in trials
        - Learn how clinical trial costs work
        - Understand informed consent
        - Learn where clinical trial care happens
        - Understand my rights as a participant
        - Avoid overly specific titles that focus on a single question, such as:
        - Learn how my doctor stays involved
        - Find out whether I'll get a placebo
        - Know when I can leave a trial
        - description should be one concise sentence explaining what the participant would learn or discuss under that goal.
        - The description should expand on the goal, not simply restate the title.
        - confidence should reflect how strongly the participant's pre-survey responses support recommending this goal relative to the other additional goals.
        - reason should explicitly reference the participant's incorrect, unsure, or high-concern responses.
        - Do not include any text outside the JSON.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You return only valid JSON. No markdown.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content.strip()
    return normalize_goal_response(json.loads(content))