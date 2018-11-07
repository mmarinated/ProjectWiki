def parse_string(string):    
    if string.find(".wikipedia.org_") > 0:
        name, agent = string.split(".wikipedia.org_")
        site = "wikipedia"
    elif string.find(".mediawiki.org_") > 0:
        name, agent = string.split(".mediawiki.org_")
        site = "mediawiki"
    elif string.find(".wikimedia.org_") > 0:
        name, agent = string.split(".wikimedia.org_")
        site = "wikimedia"
    else:
        print(string)
        raise Exception()
        
    idx = name[::-1].find('_')
    language = name[-idx:]
    name = name[:idx]
        
    idx = agent.find('_')
    access = agent[:idx]
    agent = agent[(idx+1):]
      
    return name, language, site, access, agent