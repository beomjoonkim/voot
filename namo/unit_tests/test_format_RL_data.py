from data_load_utils import *
import unittest

def find_action_in_formatted_data( a,actions ):
  for idx,a_ in enumerate(actions):
    if np.all(a.squeeze()==a_.squeeze()):
      return idx


class TestFormatRLdata( unittest.TestCase ):
  def test_sumR(self):
    traj_data = load_raw_traj_data()
    pick_data  = format_RL_data(traj_data,'pick',100)
    place_data  = format_RL_data(traj_data,'place',100)
    _,_,_,_,pick_actions,_,pick_sumR = get_sars_data( pick_data )
    _,_,_,_,place_actions,_,place_sumR = get_sars_data( place_data )

    for i in range(len(traj_data)):
      traj      = traj_data[0]
      traj_rwds = traj['r']
      true_sumR = [np.sum(traj['r'][idx:]) for idx,r in enumerate(traj['r']) ] 

      curr_sumR_to_check = 0
      for a in traj['a']:
        if max(a.shape) == 6:
          idx = find_action_in_formatted_data( a,pick_actions )
          self.assertEqual(true_sumR[curr_sumR_to_check],pick_sumR[idx,0])
        else:
          idx = find_action_in_formatted_data( a,place_actions )
          self.assertEqual(true_sumR[curr_sumR_to_check],place_sumR[idx,0])
        curr_sumR_to_check += 1
  

if __name__ == '__main__':
  unittest.main()
