"""get and print all locations loaded in all tabs in firefox

Oswald Berthold, 2018

Based on https://raymii.org/s/snippets/Get_the_current_or_all_Firefox_tab_urls_in_Bash.html

In addition, current firefox version seems to need dejsonlz4 tool from
https://github.com/avih/dejsonlz4 as a separate preprocessing step, to
uncompress the jsonlz4.

```
dejsonlz4 ~/.mozilla/firefox/RANDOM.profile/sessionstore-backups/recovery.jsonlz4 sessionstore.json
```
"""
import json, sys, os

if len(sys.argv) < 2:
    print "Need profile name as argument"
    sys.exit(1)

# f = open("/home/username/.mozilla/firefox/RANDOM.profile/sessionstore.js", "r")
f = open("{0}/.mozilla/firefox/{1}/sessionstore.json".format(os.environ['HOME'], sys.argv[1]), 'r')
jdata = json.loads(f.read())
f.close()

for win in jdata.get("windows"):
    for tab in win.get("tabs"):
        i = tab.get("index") - 1
        print tab.get("entries")[i].get("url")
