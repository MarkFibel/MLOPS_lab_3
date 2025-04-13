pipeline {
    agent any

    stages {
        stage('Start Download') {
            steps {
                build job: "CIFAR_download"
            }
        }
        
        stage('Train') {
            steps {
                script {
                    dir('/workspace/CIFAR_download') {
                        build job: "CIFAR_train"
                    }
                }
            }
        }

        stage('Deploy and Test') {
            parallel {
                stage('Deploy') {
                    steps {
                        script {
                            dir('/workspace/CIFAR_download') {
                                build job: "CIFAR_deploy"
                            }
                        }
                    }
                }
                stage('Test') {
                    steps {
                        script {
                            dir('/workspace/CIFAR_download') {
                                build job: "CIFAR_test"
                            }
                        }
                    }
                }
            }
        }
    }
}