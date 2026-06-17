import json

log_file = r'C:\Users\rajas\.gemini\antigravity-ide\brain\70548038-dd19-4978-881c-4560fffa3cb9\.system_generated\logs\transcript.jsonl'
matches = []

for line in open(log_file, 'r', encoding='utf-8'):
    try:
        step = json.loads(line)
        if step.get('type') == 'PLANNER_RESPONSE' and 'tool_calls' in step and len(step['tool_calls']) > 0:
            tool = step['tool_calls'][0]
            if tool['name'] in ['write_to_file', 'replace_file_content']:
                if 'walkthrough.md' in str(tool['args']):
                    # The content is usually a JSON string itself because it's wrapped in quotes inside the JSON args
                    content = tool['args'].get('CodeContent', tool['args'].get('ReplacementContent', ''))
                    # the args values in the transcript are JSON strings, so we decode it again
                    try:
                        decoded = json.loads(content)
                        matches.append(decoded)
                    except:
                        pass
    except Exception as e:
        continue

for m in matches:
    if "VoiceGuard — Audio Forensics Walkthrough" in m:
        with open('walkthrough_target.md', 'w', encoding='utf-8') as f:
            f.write(m)
