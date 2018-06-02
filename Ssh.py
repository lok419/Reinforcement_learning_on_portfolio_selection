import sys
import paramiko
from glob import glob
import os


def run(local, server = 'run.py', ip='147.8.182.2', username='lok419',password='12345678',port=50888):

    # Connect to remote host
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=username, password=password,port=port)
    print('Successfully connected to %s'%(ip))

    # Setup sftp connection and transmit this script
    sftp = client.open_sftp()
    sftp.put(local,server)
    print('Successfully copied the code to server')
    sftp.close()

    # Run the transmitted script remotely without args and show its output.
    # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
    print('Now executing %s ...'%(local))
    stdin, stdout, stderr = client.exec_command('python %s'%(server), get_pty=True)

    for line in iter(stdout.readline, ""):
        print(line, end = "")
    for line in iter(stderr.readline, ""):
        print(line, end = "")

    client.exec_command('rm %s'%(server))
    client.close()

def copy(local, server, ip='147.8.182.2', username='lok419',password='12345678',port=50888):

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=username, password=password,port=port)
    print('Successfully connected to 147.8.182.2')
    sftp = client.open_sftp()
    sftp.put(local,server)
    print('Successfully copied from %s to %s'%(local,server))

    sftp.close()
    client.close()

def copyall(ip='147.8.182.2', username='lok419',password='12345678',port=50888):
    # except the data files (csv)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=username, password=password,port=port)
    print('Successfully connected to 147.8.182.2')
    sftp = client.open_sftp()
    # copy all python code
    # exclude_prefixes = ('__', '.')
    for path, dirs, files in os.walk('.'):
        for f in files:
            if path not in ['.\.idea','./.idea','.\__pycache__', './__pycache__',
                            '.\data\stock','./data/stock','.\data','./data',
                            './test','.\\test']:
                local = (path+'/'+f).replace('\\','/')
                sftp.put(local,local)
                print('copied',local)
    sftp.close()
    client.close()

def copy_data(ip='147.8.182.2', username='lok419',password='12345678',port=50888):

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=username, password=password,port=port)
    print('Successfully connected to 147.8.182.2')
    sftp = client.open_sftp()
    # copy all python code
    # exclude_prefixes = ('__', '.')
    for path, dirs, files in os.walk('.'):
        for f in files:
            if path in ['.\data\stock','./data/stock','.\data','./data']:
                local = (path+'/'+f).replace('\\','/')
                sftp.put(local,local)
                print('copied',local)
    sftp.close()
    client.close()

def get(local, server, ip='147.8.182.2', username='lok419',password='12345678',port=50888):

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=username, password=password,port=port)
    print('Successfully connected to 147.8.182.2')
    sftp = client.open_sftp()
    sftp.get(server,local)
    print('Successfully received from %s to %s'%(server,local))
    sftp.close()

    client.close()

def getall(local, server, ip='147.8.182.2', username='lok419',password='12345678',port=50888):

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=username, password=password,port=port)
    print('Successfully connected to 147.8.182.2')
    stdin, stdout, stderr = client.exec_command('cd {}; ls'.format(server))
    filelist = []
    for line in iter(stdout.readline, ""):
        if '.' in line:
            filelist.append(line[:-1])

    sftp = client.open_sftp()
    for file in filelist:
        print(file)
        sftp.get(server+'/'+file, local+'/'+file)

    print('Successfully received from %s/ to %s/'%(server,local))
    sftp.close()
    client.close()

def removeall(ip='147.8.182.2', username='lok419',password='12345678',port=50888):
    # remove all except the data files (csv)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=username, password=password,port=port)
    print('Successfully connected to 147.8.182.2')

    client.exec_command('rm -rf *.py')
    client.exec_command('rm -rf *.pickle')
    client.exec_command('rm -rf result')
    client.exec_command('rm -rf model')
    client.exec_command('mkdir model')
    client.exec_command('mkdir result')
    client.exec_command('mkdir ticker')
    print('Successfully removed all files')

    client.close()

def remove_data(ip='147.8.182.2', username='lok419',password='12345678',port=50888):
    # remove all data files
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=username, password=password,port=port)
    print('Successfully connected to 147.8.182.2')

    client.exec_command('rm -rf data')
    client.exec_command('mkdir data')
    client.exec_command('cd data; mkdir stock')
    print('Successfully removed all data files')

    client.close()

def connect(ip='147.8.182.2', username='lok419',password='12345678',port=50888):

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=username, password=password,port=port)
    print('Successfully connected to 147.8.182.2')

    cd_path = []

    while True:
        cmd = input("->  ")

        if cmd == 'exit':
            client.close()
            sys.exit(0)

        if 'cd' in cmd.split(' '):
            cd_path.append('cd')
            cd_path.append( cmd.split(' ')[-1] )
            cd_path.append(';')

        else:
            stdin, stdout, stderr = client.exec_command(' '.join([' '.join(cd_path),cmd]))
            for line in iter(stdout.readline, ""):
                print('... ', line, end="")
            for line in iter(stderr.readline, ""):
                print('... ', line, end="")


if __name__ == "__main__":
    # local path: C:\Users\Cheung\PycharmProjects\DDPG\
    # server path: home\lok419\

    copy(local = 'Stock_prediction.py', server = 'Stock_prediction.py')



