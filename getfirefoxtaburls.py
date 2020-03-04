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
import json, sys, os, argparse
import pprint as pp

def main(args):
    if args.file is None:
        print('Need to provide session file with -f sessionfile.json, exiting')
        sys.exit(1)

    if not os.path.exists(args.file):
        print('File doesnt exist {0}'.format(args.file))
        sys.exit(1)

    # f = open("/home/username/.mozilla/firefox/RANDOM.profile/sessionstore.js", "r")
    # f = open("{0}/.mozilla/firefox/{1}/sessionstore.json".format(os.environ['HOME'], sys.argv[1]), 'r')
    f = open(args.file, 'r')
    jdata = json.loads(f.read())
    f.close()

    # print('tabs\n{0}'.format(pp.pformat(jdata)))
    
    for win_i, win in enumerate(jdata.get("windows")):
        # print('window {0}'.format(win))
        for tab_i, tab in enumerate(win.get("tabs")):
            # print('window {0}, tab {1}'.format(win_i, tab_i))
            i = tab.get("index") - 1
            tabentries = tab.get("entries")
            # print('    tabentries {0}'.format(len(tabentries)))
            for tabentry in tabentries:
                # print('        tabentry {0}'.format(tabentry))
                pass

            try:
                # tabentry = tabentries[i]
                tabentry = tabentries[-1]
                # print(.get("url"))
                # .get("url")
                print('    {2:03d} - {0} - {1}'.format(tabentry.get('title'), tabentry.get('url'), tab_i))
            except Exception as e:
                print('tabentries failed with {0}'.format(e))
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', type=str, help='Firefox session file made with dejsonlz4 from recovery.jsonlz4', default=None)

    args = parser.parse_args()

    main(args)

