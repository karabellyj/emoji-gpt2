import numpy as np
import scipy.stats as ss
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm


def tokenize(X, tokenizer, block_size=512):
    return tokenizer.batch_encode_plus(X, add_special_tokens=True, max_length=block_size)["input_ids"]


def predict_mean(X_ids, model, tokenizer, map_target_to_token_id):
    preds = []
    for x in tqdm(X_ids):
        outputs = model(torch.tensor(x).unsqueeze(0))
        logits = outputs[0].squeeze(0)

        target_id = logits.mean(dim=0).argmax().item()
        token_id = map_target_to_token_id[target_id]
        token = tokenizer.decode(token_id)

        preds.append(token)
    return preds


def predict_last(X_ids, model, tokenizer, map_target_to_token_id):
    preds = []
    for x in tqdm(X_ids):
        outputs = model(torch.tensor(x).unsqueeze(0))
        logits = outputs[0].squeeze(0)

        target_id = logits[-1].argmax().item()
        token_id = map_target_to_token_id[target_id]
        token = tokenizer.decode(token_id)

        preds.append(token)
    return preds


def predict_max(X_ids, model, tokenizer, map_target_to_token_id):
    preds = []
    for x in tqdm(X_ids):
        outputs = model(torch.tensor(x).unsqueeze(0))
        logits = outputs[0].squeeze(0)

        target_id = logits.max(0)[0].argmax().item()
        token_id = map_target_to_token_id[target_id]
        token = tokenizer.decode(token_id)

        preds.append(token)
    return preds


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# Entropy
def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count / len(Y)
    en = np.sum((-1) * prob * np.log2(prob))
    return en


# Joint Entropy
def joint_entropy(Y, X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y, X]
    return entropy(YX)


# Conditional Entropy
def conditional_entropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return joint_entropy(Y, X) - entropy(X)


# Uncertainty Coefficient
def theil_u(X, Y):
    """
    Reference: https://en.wikipedia.org/wiki/Uncertainty_coefficient
    """
    h_X = entropy(X)
    h_XY = conditional_entropy(X, Y)
    return (h_X - h_XY) / h_X


def train_with_reg_cv(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l2',
                      C=2 ** np.arange(-8, 1).astype(np.float), seed=42, solver='lbfgs', max_iter=int(1e6),
                      metrics='score'):
    scores = []
    for i, c in enumerate(C):
        model = LogisticRegression(C=c, penalty=penalty, random_state=seed + i, solver=solver, max_iter=max_iter)
        model.fit(trX, trY)
        if metrics == 'f1':
            y_pred = model.predict(vaX)
            score = f1_score(vaY, y_pred, average='macro')
        else:
            score = model.score(vaX, vaY)
        scores.append(score)
    c = C[np.argmax(scores)]
    model = LogisticRegression(C=c, penalty=penalty, random_state=seed + len(C), solver=solver, max_iter=max_iter)
    model.fit(trX, trY)

    return model, c, scores


def transform_last(X, model, tokenizer, args):
    X_ids = tokenize(X, tokenizer, args.block_size)
    probas = []
    for x in tqdm(X_ids):
        outputs = model(torch.tensor(x).unsqueeze(0))
        logits = outputs[0].squeeze(0)

        probas.append(logits[-1].tolist())
    return probas


def transform_mean(X, model, tokenizer, args):
    X_ids = tokenize(X, tokenizer, args.block_size)
    probas = []
    for x in tqdm(X_ids):
        outputs = model(torch.tensor(x).unsqueeze(0))
        logits = outputs[0].squeeze(0)

        probas.append(logits.mean(0).tolist())
    return probas


def transform_max(X, model, tokenizer, args):
    X_ids = tokenize(X, tokenizer, args.block_size)
    probas = []
    for x in tqdm(X_ids):
        outputs = model(torch.tensor(x).unsqueeze(0))
        logits = outputs[0].squeeze(0)

        probas.append(logits.max(0)[0].tolist())
    return probas
