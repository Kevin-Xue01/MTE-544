from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', 'decisions.py'],
            name='decisions',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', 'noisy_odom.py'],
            name='noisy_odom',
            output='screen'
        )
    ])
