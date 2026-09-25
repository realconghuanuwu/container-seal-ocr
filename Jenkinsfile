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
                echo 'Ensuring Git LFS model weights...'
                withCredentials([usernamePassword(credentialsId: 'git-token-cred', usernameVariable: 'GIT_USER', passwordVariable: 'GIT_PASS')]) {
                    sh '''
                        # If git-lfs is missing in system PATH, auto-download standalone binary for Linux
                        if ! command -v git-lfs >/dev/null 2>&1; then
                            if [ ! -f .git-lfs-bin/git-lfs ]; then
                                echo "Notice: git-lfs not found in system PATH. Auto-downloading standalone git-lfs..."
                                mkdir -p .git-lfs-bin
                                curl -sL https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz | tar -xz --strip-components=1 -C .git-lfs-bin
                                chmod +x .git-lfs-bin/git-lfs
                            fi
                            export PATH="$PWD/.git-lfs-bin:$PATH"
                        fi

                        echo "Using Git LFS version: $(git lfs version 2>/dev/null || git-lfs version)"
                        git-lfs install --local 2>/dev/null || git lfs install --local 2>/dev/null || true

                        # Configure authenticated remote for private Git LFS pull
                        git config remote.origin.url "https://${GIT_USER}:${GIT_PASS}@github.com/realconghuanuwu/container-seal-ocr.git"

                        echo "Pulling Git LFS binary model files..."
                        git-lfs pull || git lfs pull

                        # Revert remote URL to keep repo clean
                        git config remote.origin.url "https://github.com/realconghuanuwu/container-seal-ocr.git"

                        # Sanity check model weight files are not small LFS pointers
                        if [ -f models/seal-det-v1.0.0/best.onnx ]; then
                            SIZE=$(wc -c < models/seal-det-v1.0.0/best.onnx)
                            echo "Verified models/seal-det-v1.0.0/best.onnx size: $SIZE bytes"
                            if [ "$SIZE" -lt 1000 ]; then
                                echo "ERROR: models/seal-det-v1.0.0/best.onnx is an un-downloaded Git LFS pointer ($SIZE bytes)!"
                                exit 1
                            fi
                        fi
                    '''
                }
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
