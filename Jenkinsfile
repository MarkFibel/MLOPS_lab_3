pipeline {
    agent any

    stages {
        stage('Clone repo') {
            steps {
                git 'https://github.com/username/your-ml-project.git'
            }
        }
        stage('Install dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Download data') {
            steps {
                sh 'python scripts/download_data.py'
            }
        }
        stage('Preprocess data') {
            steps {
                sh 'python scripts/preprocess.py'
            }
        }
        stage('Train model') {
            steps {
                sh 'python scripts/train.py'
            }
        }
        stage('Deploy model') {
            steps {
                sh 'nohup python scripts/serve.py &'
            }
        }
    }
}
