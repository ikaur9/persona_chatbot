# persona_chatbot
Final project for W266

**MODEL TYPE:**

- baseline
- baseline_named
- triplehead

**DATA SIZE:**

- 10
- 100

To train a model, run the following commands:

`> pip install -r requirements.txt`

`> cd model_[MODEL TYPE]`

`> python train_[MODEL TYPE]_[DATA SIZE].py [--args]`

To test a pre-trained model, run the following command in the same directory:

`> python train_[MODEL TYPE]_[DATA SIZE].py --model_checkpoint [CHECKPOINT FOLDER] --test --n_epochs 0 [--args]`

To interact with a pre-trained model, run the following command in the same directory:

`> python interact_[MODEL TYPE]_[DATA SIZE].py --model_checkpoint [CHECKPOINT FOLDER] [--args]`