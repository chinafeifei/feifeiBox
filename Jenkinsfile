pipeline {
  agent {
    docker {
      image 'node:6-alpine'
      args '-p 3000:3000'
    }

  }
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
        sh 'cat test.sh'
      }
    }

  }
  environment {
    CI = 'true'
  }
}