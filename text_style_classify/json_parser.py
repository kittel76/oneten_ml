import json



def getText(jsonData):
    data = json.loads(jsonData)

    text = ''

    for idx in range(len(data)):
        if data[idx]['type'] == 'text':
            if(isTargetText(text)==True):
                print("text", text)
                text += ' ' + data[idx]['value']

    return text


def isSkipText(text):
    if (text.find("MD COMMENT")):
        return True;
    return False


def isTargetText(text):
    if(text.find("MD COMMENT") >= 0):
        return True
    if(text.find("Stories") >= 0):
        return True

    return False





