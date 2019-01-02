import os
import time

while True:
  tmp_files = os.listdir('/tmp/')
  for f in tmp_files:
    temp_file_path = '/tmp/'+f+'/pr2-beta-static.dae'
    if os.path.exists(temp_file_path):
      command = 'rm -rf /tmp/'+f
      while time.time() - os.stat(temp_file_path).st_ctime <= 10:
        time.sleep(1)
      print temp_file_path
      os.system(command)
