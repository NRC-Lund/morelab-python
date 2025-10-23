"""
Test MySQL database connection using environment variables or command line args.
"""
import os
import sys

try:
    import mysql.connector
except ImportError:
    print("Missing dependency: mysql-connector-python. Install with: pip install mysql-connector-python", file=sys.stderr)
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Test MySQL database connection.")
    parser.add_argument('--host', default=os.environ.get('DB_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('DB_PORT', '3306')))
    parser.add_argument('--user', default=os.environ.get('DB_USER', 'root'))
    parser.add_argument('--password', default=os.environ.get('DB_PASSWORD', ''))
    parser.add_argument('--database', default=os.environ.get('DB_NAME', 'Hipster'))

    # SSH tunneling arguments
    parser.add_argument('--ssh-host', default=os.environ.get('SSH_HOST'), help='SSH host for tunneling')
    parser.add_argument('--ssh-port', type=int, default=int(os.environ.get('SSH_PORT', '22')), help='SSH port')
    parser.add_argument('--ssh-user', default=os.environ.get('SSH_USER'), help='SSH username')
    parser.add_argument('--ssh-key', default=os.environ.get('SSH_KEY', os.path.expanduser('~/.ssh/id_rsa')), help='SSH private key path')
    parser.add_argument('--ssh-passphrase', default=os.environ.get('SSH_PASSPHRASE'), help='SSH key passphrase (optional)')

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    tunnel = None
    # SSH options
    ssh_host = args.ssh_host
    ssh_port = args.ssh_port
    ssh_user = args.ssh_user
    ssh_key = args.ssh_key
    ssh_passphrase = args.ssh_passphrase

    try:
        if ssh_host:
            try:
                from sshtunnel import SSHTunnelForwarder
            except ImportError:
                print("Missing dependency: sshtunnel. Install with: pip install sshtunnel", file=sys.stderr)
                sys.exit(1)

            tunnel = SSHTunnelForwarder(
                (ssh_host, ssh_port),
                ssh_username=ssh_user,
                ssh_pkey=ssh_key,
                ssh_private_key_password=ssh_passphrase,
                remote_bind_address=("127.0.0.1", 3306),
                local_bind_address=("127.0.0.1", 0),
            )
            tunnel.start()
            db_host = "127.0.0.1"
            db_port = int(tunnel.local_bind_port)
        else:
            db_host = args.host
            db_port = args.port

        conn = mysql.connector.connect(
            host=db_host,
            port=db_port,
            user=args.user,
            password=args.password,
            database=args.database
        )
        print("Connection successful!")
        conn.close()
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tunnel:
            tunnel.stop()

if __name__ == "__main__":
    main()
