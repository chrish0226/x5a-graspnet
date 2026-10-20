from setuptools import setup
from glob import glob
package_name='x5a_handeye'
setup(
  name=package_name,
  version='0.1.0',
  packages=[package_name],
  data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/config', glob('config/*')),
    ('share/' + package_name + '/launch', glob('launch/*.py')),
  ],
  install_requires=['setuptools'],
  zip_safe=True,
  maintainer='x5a user',
  maintainer_email='user@example.com',
  description='X5A eye-in-hand and eye-to-hand calibration',
  license='BSD',
  entry_points={'console_scripts': [
    'board_detector = x5a_handeye.board_detector:main',
    'sample_collector = x5a_handeye.sample_collector:main',
    'solve_handeye = x5a_handeye.solve_handeye:main',
    'refine_robot_model = x5a_handeye.refine_robot_model:main',
    'recalibrate_fixed_model = x5a_handeye.recalibrate_fixed_model:main',
    'check_model_candidate = x5a_handeye.check_model_candidate:main',
    'readonly_joint_states = x5a_handeye.readonly_joint_states:main',
    'validate_calibration = x5a_handeye.validate_calibration:main',
    'publish_handeye_tf = x5a_handeye.publish_handeye_tf:main',
    'gravity_teach = x5a_handeye.gravity_teach:main',
    'readonly_handeye_sampler = x5a_handeye.readonly_handeye_sampler:main',
    'reframe_samples = x5a_handeye.reframe_samples:main',
  ]},
)
