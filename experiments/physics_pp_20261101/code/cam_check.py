import sys, numpy as np
sys.path.insert(0, '/rhome/tanuj/home_moved_20261029/V2D_challenge/experiments/physics_pp_20261101/code')
import physics_pp_v2 as pp
IN = '/rhome/tanuj/home_moved_20261029/V2D_challenge/experiments/joint_kaggle_20261031/ready_v2/track1_pi3x_joint_physics.parquet'
for e in (0, 5, 11, 18, 22, 29):
    ep = pp.load_episode(IN, e)
    v, j = pp.decode(ep['pose'][:3], ep['scales'], ep['shape'])
    print(e, 'obj t0', np.round(ep['t'][0], 2), 'human centroid', np.round(v[0].mean(0), 2), 'obj scale', round(ep['s'], 3),
          'mesh extent', np.round(ep['verts'].max(0) - ep['verts'].min(0), 2), 'mesh centre', np.round(0.5 * (ep['verts'].max(0) + ep['verts'].min(0)), 2))
