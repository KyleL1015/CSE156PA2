import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import ClassifierModel, TransformerDecoder
from utilities import Utilities
import torch.nn as nn


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoder_model, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoder on the data in data_loader."""
    decoder_model.eval()
    losses = []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            logits, _ = decoder_model(X)  # [B, T, vocab_size]
            # Reshape for loss computation
            logits_flat = logits.view(-1, logits.size(-1))  # [B*T, vocab_size]
            Y_flat = Y.view(-1)  # [B*T]
            loss = nn.CrossEntropyLoss()(logits_flat, Y_flat)
            losses.append(loss.item())
            if len(losses) >= eval_iters:
                break
    
    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()
    decoder_model.train()
    return perplexity

def main(part='both'):

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=False)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

     # for the classification  task, you will train for a fixed number of epochs like this:

    # Part 1: Encoder + classifier (optional)
    if part in ('part1', 'both'):
        # instantiate classifier model, optimizer and loss
        model = ClassifierModel(tokenizer.vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, n_hidden=n_hidden, n_output=n_output, max_len=block_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        encoder_param_count = sum(p.numel() for p in model.encoder.parameters())
        print(f"Encoder parameter count: {encoder_param_count}")

        # classification training
        for epoch in range(epochs_CLS):
            running_loss = 0.0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_acc = compute_classifier_accuracy(model, train_CLS_loader)
            test_acc = compute_classifier_accuracy(model, test_CLS_loader)
            print(f"Epoch {epoch+1}/{epochs_CLS}: loss={running_loss:.4f}, train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%")

        print("\n=== Final Test Accuracy ===")
        final_test_acc = compute_classifier_accuracy(model, test_CLS_loader)
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")

        print("\n=== Sanity Check: Attention Weights ===")
        # Test attention with a sample sentence
        sample_sentences = [
            "The economy is growing strong",
            "We must act now on climate change",
            "The future is bright for America"
        ]
        utilities = Utilities(tokenizer, model.encoder)
        for sentence in sample_sentences:
            print(f"\nChecking attention for: '{sentence}'")
            utilities.sanity_check(sentence, block_size)

    # Part 2: Decoder LM (optional)
    if part in ('part2', 'both'):
        print("\n\n=== PART 2: Decoder Language Modeling ===")
        
        # Load test datasets for LM evaluation
        with open('speechesdataset/test_LM_obama.txt', 'r', encoding='utf-8') as f:
            obama_text = f.read()
        with open('speechesdataset/test_LM_wbush.txt', 'r', encoding='utf-8') as f:
            wbush_text = f.read()
        with open('speechesdataset/test_LM_hbush.txt', 'r', encoding='utf-8') as f:
            hbush_text = f.read()
        
        test_obama_dataset = LanguageModelingDataset(tokenizer, obama_text, block_size)
        test_wbush_dataset = LanguageModelingDataset(tokenizer, wbush_text, block_size)
        test_hbush_dataset = LanguageModelingDataset(tokenizer, hbush_text, block_size)
        
        test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=False)
        test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=False)
        test_hbush_loader = DataLoader(test_hbush_dataset, batch_size=batch_size, shuffle=False)
        
        # Instantiate decoder
        decoder = TransformerDecoder(tokenizer.vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, max_len=block_size).to(device)
        optimizer_lm = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion_lm = nn.CrossEntropyLoss()
        
        decoder_param_count = sum(p.numel() for p in decoder.parameters())
        print(f"Decoder parameter count: {decoder_param_count}")
        
        # Decoder training loop
        print("\nTraining decoder...")
        perplexity_history = []
        
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            
            xb, yb = xb.to(device), yb.to(device)
            optimizer_lm.zero_grad()
            
            logits, _ = decoder(xb)  # [B, T, vocab_size]
            logits_flat = logits.view(-1, logits.size(-1))
            yb_flat = yb.view(-1)
            loss = criterion_lm(logits_flat, yb_flat)
            
            loss.backward()
            optimizer_lm.step()
            
            # Evaluate every eval_interval iterations
            if (i + 1) % eval_interval == 0:
                perp = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
                perplexity_history.append(perp)
                print(f"Iteration {i+1}/{max_iters}: train_perplexity={perp:.2f}")
        
        print(f"\nFinal iteration {max_iters} completed")
        
        # Final evaluation on train set
        final_perp = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
        print(f"Final train perplexity: {final_perp:.2f}")
        
        # Evaluate on test sets
        print("\n=== Test Set Perplexities ===")
        obama_perp = compute_perplexity(decoder, test_obama_loader, eval_iters=eval_iters)
        print(f"Obama test perplexity: {obama_perp:.2f}")
        
        wbush_perp = compute_perplexity(decoder, test_wbush_loader, eval_iters=eval_iters)
        print(f"W. Bush test perplexity: {wbush_perp:.2f}")
        
        hbush_perp = compute_perplexity(decoder, test_hbush_loader, eval_iters=eval_iters)
        print(f"H. Bush test perplexity: {hbush_perp:.2f}")
        
        print("\n=== Sanity Check: Decoder Attention (using Utilities) ===")
        sample_sentences = [
            "the president spoke about the future",
            "we must act now to protect our planet"
        ]
        utilities_decoder = Utilities(tokenizer, decoder)
        for s in sample_sentences:
            print(f"\nChecking decoder attention for: '{s}'")
            utilities_decoder.sanity_check(s, block_size)

        print("\nPart 2 complete!")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-part', choices=['part1', 'part2', 'both'], default='both')
    args = parser.parse_args()
    main(part=args.part)
