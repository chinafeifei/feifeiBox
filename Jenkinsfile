pipeline {
  agent any
  stages {
    stage('Build') {
      steps {
        echo 'We will run test script'
      }
    }

    stage('Test') {
      steps {
        sh 'ls'
        sh 'chmod +x test.sh'
        sh 'bash ./test.sh '
      }
    }

  }
  environment {
    CI = 'true'
  }
}