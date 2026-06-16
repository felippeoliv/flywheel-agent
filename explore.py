import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flywheel.appworld_env import AppWorldEnv
tid = sys.argv[1] if len(sys.argv) > 1 else "50e1ac9_1"
env = AppWorldEnv(tid, experiment_name="explore")
code = '''
me = apis.supervisor.show_profile()
pw = {p['account_name']: p['password'] for p in apis.supervisor.show_account_passwords()}
tok = apis.spotify.login(username=me['email'], password=pw['spotify'])['access_token']
sids=set()
pi=0
while True:
    r=apis.spotify.show_song_library(access_token=tok,page_index=pi,page_limit=20)
    if not r: break
    for s in r: sids.add(s['song_id'])
    pi+=1
pi=0
while True:
    r=apis.spotify.show_album_library(access_token=tok,page_index=pi,page_limit=20)
    if not r: break
    for a in r: sids.update(a.get('song_ids',[]))
    pi+=1
pi=0
while True:
    r=apis.spotify.show_playlist_library(access_token=tok,page_index=pi,page_limit=20)
    if not r: break
    for p in r: sids.update(p.get('song_ids',[]))
    pi+=1
from collections import Counter
genres=Counter()
rb=[]
for sid in sids:
    s=apis.spotify.show_song(song_id=sid)
    genres[s.get('genre')]+=1
    if str(s.get('genre','')).strip().lower() in ('r&b','r&b','rnb','r&b/soul'):
        rb.append((s['title'], s.get('play_count')))
print("total songs:", len(sids))
print("distinct genres:", dict(genres))
rb.sort(key=lambda x:(x[1] or 0), reverse=True)
print("r&b songs sorted:", rb[:8])
'''
print(env.world.execute(code))
env.close()
