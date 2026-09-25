pipeline {
    agent any

    environment {
        IMAGE_NAME = 'container-seal-ocr'
        CONTAINER_NAME = 'container-seal-ocr'
        // Port dị: 17712 (chuẩn ISO 17712 về Container Mechanical Seal, không bao giờ trùng trên server lab)
        HOST_PORT = '17712'
        CONTAINER_PORT = '7860'
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout & Git LFS') {
            steps {
                echo 'Checking out source code and ensuring Git LFS model weights...'
                checkout scm
                sh '''
                    # Pull Git LFS objects if git-lfs is available on agent
                    if git lfs version >/dev/null 2>&1; then
                        echo "Git LFS detected, pulling model binary files..."
                        git lfs pull
                    else
                        echo "Notice: git-lfs command not in PATH, verifying model files exist..."
                    fi

                    # Sanity check model weight files are not small LFS pointers
                    if [ -f models/seal-det-v1.0.0/best.onnx ]; then
                        SIZE=$(wc -c < models/seal-det-v1.0.0/best.onnx)
                        if [ "$SIZE" -lt 1000 ]; then
                            echo "ERROR: models/seal-det-v1.0.0/best.onnx is an un-downloaded Git LFS pointer ($SIZE bytes)!"
                            echo "Please install git-lfs on the Jenkins node or run 'git lfs pull'."
                            exit 1
                        fi
                    fi
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image: ${IMAGE_NAME}:${BUILD_NUMBER} & ${IMAGE_NAME}:latest..."
                sh """
                    docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} -t ${IMAGE_NAME}:latest .
                """
            }
        }

        stage('Deploy to Lab Server') {
            steps {
                echo "Deploying container ${CONTAINER_NAME} on port ${HOST_PORT}..."
                sh """
                    # Stop & remove old container if running
                    docker stop ${CONTAINER_NAME} >/dev/null 2>&1 || true
                    docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true

                    # Run new container
                    docker run -d \\
                        --name ${CONTAINER_NAME} \\
                        --restart unless-stopped \\
                        -p ${HOST_PORT}:${CONTAINER_PORT} \\
                        ${IMAGE_NAME}:latest
                """
            }
        }

        stage('Verify Health Check') {
            steps {
                echo "Verifying service readiness at http://localhost:${HOST_PORT}/health/ready..."
                sh """
                    # Poll up to 60 seconds for PaddleOCR and YOLO models to initialize
                    for i in \$(seq 1 12); do
                        echo "Readiness probe attempt \$i/12..."
                        STATUS=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${HOST_PORT}/health/ready || true)
                        if [ "\$STATUS" = "200" ]; then
                            echo "SUCCESS: Service is UP and healthy!"
                            exit 0
                        fi
                        sleep 5
                    done
                    echo "ERROR: Readiness check timed out after 60 seconds."
                    docker logs --tail 60 ${CONTAINER_NAME}
                    exit 1
                """
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment Successful! Open http://<lab-server-ip>:${HOST_PORT}/test to test."
        }
        failure {
            echo "❌ Pipeline Failed! Check build log and container status."
        }
        always {
            sh '''
                # Clean up dangling images to keep lab server disk clean
                docker image prune -f >/dev/null 2>&1 || true
            '''
        }
    }
}
