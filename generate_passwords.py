# generate_passwords.py

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import bcrypt

def generate_passwords():
    """Generate hashed passwords and create config file"""
    credentials = {
        'usernames': {
            'admin': {
                'email': 'pickengine@gmail.com',
                'name': 'Admin User',
                'password': bcrypt.hashpw('admin123'.encode(), bcrypt.gensalt()).decode()
            },
            'analyst': {
                'email': 'shiv@dev.com',
                'name': 'Analyst User',
                'password': bcrypt.hashpw('Shiv0518'.encode(), bcrypt.gensalt()).decode()
            }
        }
    }

    config = {
        'credentials': credentials,
        'cookie': {
            'expiry_days': 30,
            'key': 'nba_predictions_signature_key',
            'name': 'nba_predictions_auth'
        }
    }

    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


    print("\nGenerated config.yaml with hashed passwords:")
    print("\nUsernames and their hashed passwords:")
    for username, data in credentials['usernames'].items():
        print(f"\nUsername: {username}")
        print(f"Hashed password: {data['password']}")

if __name__ == "__main__":
    generate_passwords()
    print("\nConfiguration file generated successfully!")


