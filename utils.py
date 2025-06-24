def clean_and_flatten_skills(skill_dict):
    """
    Converts a skill dictionary like:
    {"Python": 8, "SQL": 6, "Git": 5}
    into a lowercase, comma-separated string: "python, sql, git"
    """
    return ", ".join([skill.lower().strip() for skill in skill_dict.keys()])
