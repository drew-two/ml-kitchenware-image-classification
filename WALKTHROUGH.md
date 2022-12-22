# Walkthrough

1. Running `make repo`
    ![Repo Creation](./assets/images/make-aws-1.PNG)
2. Running `make publish`
    ![Docker push](./assets/images/make-aws-2.PNG)
3. Running `make aws`
    ![VPC Stack](./assets/images/cloudformation-1.PNG)
    ![ECS Cluster Stack](./assets/images/cloudformation-2.PNG)
    ![ECS Task Stack](./assets/images/cloudformation-3.PNG)
    ![Cloudformation Finished](./assets/images/cloudformation-4.PNG)
4. Move to AWS and go to ECR. Go to Task Definitions. Click the one like 'kitchenware-classification-ecs-task-definition-ECSTaskDefinition'
    ![ECR](./assets/images/task-definition-1.png)
5. Click on the latest revision of the Task definition
    ![Task Definition Revision](./assets/images/task-definition-2.png)
6. Click Actions > Run Task
    ![Run Task](./assets/images/task-definition-3.png)
7. Choose the settings in the image. Click the new VPC
    ![Settings](./assets/images/task-definition-4.png)
8. Edit the security group and open TCP port 3000
    ![Security Group](./assets/images/task-definition-5.png)
9. Make sure the IP will auto-assign and Run Task
    ![Finish Task](./assets/images/task-definition-6.png)
10. Go to the new Task and note the public IP. Wait for the status to say `RUNNING`
    ![Finish Task](./assets/images/task-definition-7.png)
11. Go to the public IP at port 3000 in your web browser (e.g. http://xxx.xxx.xxx.xxx:3000). Click on the POST API
    ![BentoML](./assets/images/task-definition-8.png)
12. Click `Try it out` and set the file type to JPEG image.
    ![Prepare Request](./assets/images/task-definition-9.png)
13. Upload the image you want to predict on and hit `Execute`. You will see the prediction in the response body.
    ![Response](./assets/images/task-definition-10.png)