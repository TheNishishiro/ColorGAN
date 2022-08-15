# ColorGAN

Welcome to my ColorGAN project!

Inspired by a lack of similar projects (well, there was a lack of similar projects back in 2019), I decided to create a ColorGAN which is a very simple GAN that can be used to generate colored images.

My goal was to have a simple implementation that anyone can use!

Hopefully I can extend it into more use cases as well while still keeping it beginner friendly.

## 2022 Update

I recently picked this project back up again, wanting to make use of my new GPU and to blow some dust off my ML experience.

So I decided to clean up the code and retrain it from scratch.

**ColorGAN_Legacy** folder contains the old version of the code and well as the Hentai dataset used back then, unfortunately I lost models for that task so they are no longer available.

**ColorGAN 2** folder contains the new version of the code, it's a bit cleaner in my opinion and uses the new TF 2.x which has Keras built into it. Documentation is still under development but it's not much different from previous implementation.

**ColorGanInterface** is a .NET win forms application that can communicate with ColorGAN REST API to upload a gray scaled image and display a colored one.

## Things to do

I don't have much of a free time, but I would like to retrain this model for different anime related applications