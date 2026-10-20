"""Keep ROS tests on a separate DDS domain from the running physical arm."""
import os

os.environ['ROS_DOMAIN_ID'] = os.environ.get('X5A_TEST_ROS_DOMAIN_ID', '149')
