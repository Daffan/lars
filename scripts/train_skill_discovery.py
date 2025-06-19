import argparse
import random
import os
import datetime
import json

# add parent path to sys
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.algos.skill_discovery import Encoder, Decoder, Model, get_batched_loss
from src.data import data_loader_dict
from src.envs.encoders import BERTEncoder, OpenAIEncoder
from src.utils.utils import compute_embeddings

def sample_ids_from_clustering(embeddings, cluster_centers, num_examples=3):
    # compute distance to cluster center
    distances = np.linalg.norm(embeddings[None, :, :] - cluster_centers[:, None, :], axis=-1)
    return np.argsort(distances, axis=1)[:, :num_examples]

def parse_args():
    parser = argparse.ArgumentParser()
    # user config
    parser.add_argument('--seed', type=int, default=None)
    # task config
    parser.add_argument('--task', type=str, default='tabmwp')
    parser.add_argument('--split', type=str, default='train_remain')  # those two are the test splits
    # prompting config
    parser.add_argument('--encoder', type=str, default='microsoft/deberta-v2-xlarge')
    parser.add_argument('--prompt_format', type=str, default='CQ-S', choices=["CQ-S", "Q-S", "CQP-N"], help='format of the exmaples to prompt the LLM')
    # training config
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epoch', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_layer', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--qs_as_whole', action='store_true', help='if true, qs is treated as a whole')
    parser.add_argument('--reconstruct_qs', action='store_true', help='if true, reconstruct qs, otherwise only reconstruct s')
    parser.add_argument('--kl_ratio', type=float, default=1.0, help='weight of kl loss')
    parser.add_argument('--clip', type=float, default=10., help='clip the policy gradient')
    # test config
    parser.add_argument('--test', action='store_true', help='if true, only test the model')
    parser.add_argument('--test_split', type=str, default='train_cand1k')
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--plot_model', type=str, default="tsne", choices=["tsne", "pca"], help="which model to plot the embeddings")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seeding
    if args.seed is None:
        seed = random.randint(0, 10000)
    else:
        seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = vars(args)
    cfg["seed"] = seed
    
    ## Initialization
    # load the dataset
    assert args.task in data_loader_dict.keys(), f"Invalid task {args.task}"
    train_dataset = data_loader_dict[args.task](split=args.split, shuffle=False)

    # load encoder
    if args.encoder in ["text-embedding-ada-002", "code-search-babbage-text-001", "code-search-babbage-code-001"]:
        encoder = OpenAIEncoder(args.encoder)
    else:
        encoder = BERTEncoder(args.encoder, device=device)

    # load embeddings
    # train_qs_embeddings = compute_embeddings(args.task, args.split, args.encoder, "CQ-S", encoder, test=False)
    q_format = args.prompt_format.split("-")[0]
    s_format = args.prompt_format.split("-")[1]
    train_q_embeddings, (_, correct_label) = compute_embeddings(args.task, args.split, args.encoder, q_format, encoder, test=False)
    correct_label = torch.from_numpy(correct_label).to(device)
    if args.qs_as_whole:
        train_qs_embeddings, _ = compute_embeddings(args.task, args.split, args.encoder, args.prompt_format, encoder, test=False)
    if args.reconstruct_qs:
        train_s_embeddings = train_qs_embeddings
    else:
        train_s_embeddings, _ = compute_embeddings(args.task, args.split, args.encoder, s_format, encoder, test=False)

    train_q_embeddings = torch.from_numpy(train_q_embeddings).to(device).float()
    train_s_embeddings = torch.from_numpy(train_s_embeddings).to(device).float()
    if args.qs_as_whole:
        train_qs_embeddings = torch.from_numpy(train_qs_embeddings).to(device).float()
    else:
        train_qs_embeddings = torch.cat([train_q_embeddings, train_s_embeddings], dim=1).float()

    # load model
    q_encoder = Encoder(
        input_dim=train_q_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        hidden_layer=args.hidden_layer,
        latent_dim=args.latent_dim).to(device)
    qs_encoder = Encoder(
        input_dim=train_qs_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        hidden_layer=args.hidden_layer,
        latent_dim=args.latent_dim).to(device)
    pi = Encoder(
        input_dim=train_q_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        hidden_layer=args.hidden_layer,
        latent_dim=args.latent_dim).to(device)
    decoder = Decoder(
        input_dim=train_q_embeddings.shape[1],
        latent_dim=args.latent_dim,
        hidden_layer=args.hidden_layer,
        hidden_dim=args.hidden_dim,
        output_dim=train_s_embeddings.shape[1]).to(device)
    model = Model(q_encoder, qs_encoder, pi, decoder, device).to(device)

    if args.load is not None:
        state_dict = torch.load(args.load)
        model.load_state_dict(state_dict)
        print("Load the model from %s" %args.load)

    if not args.test:
        file_name = os.path.basename(__file__)
        datetime_print = datetime.datetime.now().strftime("%b-%d-%Y-%H:%M:%S") + "-%s" %args.task
        save_path = os.path.join("results", file_name, datetime_print)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        writer = SummaryWriter(save_path)

        optimizer = Adam(model.parameters(), lr=args.lr)
        num_batches = len(train_q_embeddings) // args.batch_size
        for epoch in range(args.num_epoch):
            overall_loss = 0
            overall_recon_loss = 0
            overall_kl_loss = 0
            overall_pi_loss = 0
            batch_idx = torch.randperm(len(train_q_embeddings))
            for i in range(num_batches):
                if i < num_batches - 1:
                    q_batch = train_q_embeddings[batch_idx[i*args.batch_size:(i+1)*args.batch_size]]
                    s_batch = train_s_embeddings[batch_idx[i*args.batch_size:(i+1)*args.batch_size]]
                    qs_batch = train_qs_embeddings[batch_idx[i*args.batch_size:(i+1)*args.batch_size]]
                    y_batch = correct_label[batch_idx[i*args.batch_size:(i+1)*args.batch_size]]
                else:
                    q_batch = train_q_embeddings[batch_idx[i*args.batch_size:]]
                    s_batch = train_s_embeddings[batch_idx[i*args.batch_size:]]
                    qs_batch = train_qs_embeddings[batch_idx[i*args.batch_size:]]
                    y_batch = correct_label[batch_idx[i*args.batch_size:]]

                # ----------------- train the model -----------------
                optimizer.zero_grad()
                loss, (recon_loss, kl_loss, pi_loss) = get_batched_loss(q_batch, s_batch, qs_batch, y_batch, args.kl_ratio, model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.pi.parameters(), args.clip)
                optimizer.step()

                overall_loss += loss.item()
                overall_recon_loss += recon_loss.item()
                overall_kl_loss += kl_loss.item()
                overall_pi_loss += pi_loss.item()
            total_loss = overall_loss / num_batches
            total_recon_loss = overall_recon_loss / num_batches
            total_kl_loss = overall_kl_loss / num_batches
            total_pi_loss = overall_pi_loss / num_batches
            print("Epoch %.3d | Total Loss: %.2e | Reconstruction loss: %.2e | KL loss: %.2e | pi loss: %.2f" \
                %(epoch, total_loss, total_recon_loss, total_kl_loss, total_pi_loss))
            writer.add_scalar("train/total_loss", total_loss, epoch)
            writer.add_scalar("train/recon_loss", total_recon_loss, epoch)
            writer.add_scalar("train/kl_loss", total_kl_loss, epoch)
        
        # ----------------- save the model -----------------
        torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            cfg = vars(args)
            json.dump(cfg, f, indent=2, separators=(',', ': '))
        print("Save the model and config to %s" %save_path)
    del train_q_embeddings, train_s_embeddings, train_qs_embeddings





    # ----------------- evaluate the model -----------------
    from sklearn.cluster import SpectralClustering, KMeans
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import normalized_mutual_info_score
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    if args.test:
        save_path = os.path.dirname(args.load)

    ## load dataset and embeddings
    test_dataset = data_loader_dict[args.task](split=args.test_split, shuffle=False)

    test_q_embeddings_np, (q_text, _) = compute_embeddings(args.task, args.test_split, args.encoder, q_format, encoder, test=False)
    test_s_embeddings_np, (s_text, _) = compute_embeddings(args.task, args.test_split, args.encoder, s_format, encoder, test=False)
    if args.qs_as_whole:
        test_qs_embeddings_np, _ = compute_embeddings(args.task, args.test_split, args.encoder, args.prompt_format, encoder, test=False)
    else:
        test_qs_embeddings_np = np.concatenate([test_q_embeddings_np, test_s_embeddings_np], axis=1)

    # test_s_embeddings_np = np.random.permutation(test_s_embeddings_np)[:200]
    # test_q_embeddings_np = np.random.permutation(test_q_embeddings_np)[:200]
    # test_qs_embeddings_np = np.random.permutation(test_qs_embeddings_np)[:200]

    test_s_embeddings = torch.from_numpy(test_s_embeddings_np).to(device)
    test_q_embeddings = torch.from_numpy(test_q_embeddings_np).to(device)
    test_qs_embeddings = torch.from_numpy(test_qs_embeddings_np).to(device)

    ## compute skill embeddings
    # skill infered from CQ
    q_skill_embeddings = []
    qs_skill_embeddings = []
    pi_skill_embeddings = []
    num_batches = len(test_q_embeddings) // args.batch_size + 1
    for i in range(num_batches):
        if i < num_batches - 1:
            q_batch = test_q_embeddings[i*args.batch_size:(i+1)*args.batch_size]
            s_batch = test_s_embeddings[i*args.batch_size:(i+1)*args.batch_size]
            qs_batch = test_qs_embeddings[i*args.batch_size:(i+1)*args.batch_size]
        else:
            q_batch = test_q_embeddings[i*args.batch_size:]
            s_batch = test_s_embeddings[i*args.batch_size:]
            qs_batch = test_qs_embeddings[i*args.batch_size:]
        # qs_batch = torch.cat([q_batch, s_batch], dim=1)

        q_skill_embeddings.append(q_encoder(q_batch)[0].detach().cpu().numpy())
        qs_skill_embeddings.append(qs_encoder(qs_batch)[0].detach().cpu().numpy())
        pi_skill_embeddings.append(pi(q_batch)[0].detach().cpu().numpy())

    q_skill_embeddings = np.concatenate(q_skill_embeddings, axis=0)
    qs_skill_embeddings = np.concatenate(qs_skill_embeddings, axis=0)
    pi_skill_embeddings = np.concatenate(pi_skill_embeddings, axis=0)

    # ----------------- clustering -----------------
    # clustering_method = SpectralClustering
    clustering_method = KMeans

    clustering_model = clustering_method(n_clusters=args.num_clusters, random_state=args.seed)
    y_km = clustering_model.fit_predict(q_skill_embeddings)
    distances = np.linalg.norm(q_skill_embeddings[None, :, :] - clustering_model.cluster_centers_[:, None, :], axis=-1)
    q_skill_cluster_ids = np.argsort(distances, axis=1)[:, :5]

    clustering_model = clustering_method(n_clusters=args.num_clusters, random_state=args.seed)
    qs_skill_y_km = clustering_model.fit_predict(qs_skill_embeddings)
    distances = np.linalg.norm(qs_skill_embeddings[None, :, :] - clustering_model.cluster_centers_[:, None, :], axis=-1)
    qs_skill_cluster_ids = np.argsort(distances, axis=1)[:, :5]

    """ # save the labels for all instances here
    # Load data
    data_root = "assets/dataset/tablemwp"
    split = "problems_train_cand1k.json"
    data = json.load(open(os.path.join(data_root, split)))
    data_with_label = {}

    for k, y in zip(data.keys(), qs_skill_y_km):
        data_with_label[k] = data[k]
        data_with_label[k]["label"] = int(y)

    json.dump(data_with_label, open(os.path.join(data_root, "problems_train_cand1k_labeled.json"), "w"), indent=2, separators=(',', ': ')) """
    labeled_data = json.load(open(f"assets/dataset/tablemwp/problems_{args.test_split}.json"))
    instance_0 = labeled_data[list(labeled_data.keys())[0]]
    if "label" in instance_0.keys():
        ground_truth_cluster_ids = np.array([int(labeled_data[k]["label"]) for k in labeled_data.keys()])
    else:
        ground_truth_cluster_ids = None

    clustering_model = clustering_method(n_clusters=args.num_clusters, random_state=args.seed)
    q_y_km = clustering_model.fit_predict(test_q_embeddings_np)
    distances = np.linalg.norm(test_q_embeddings_np[None, :, :] - clustering_model.cluster_centers_[:, None, :], axis=-1)
    q_cluster_ids = np.argsort(distances, axis=1)[:, :5]

    clustering_model = clustering_method(n_clusters=args.num_clusters, random_state=args.seed)
    qs_y_km = clustering_model.fit_predict(test_qs_embeddings_np)
    distances = np.linalg.norm(test_qs_embeddings_np[None, :, :] - clustering_model.cluster_centers_[:, None, :], axis=-1)
    qs_cluster_ids = np.argsort(distances, axis=1)[:, :5]

    clustering_model = clustering_method(n_clusters=args.num_clusters, random_state=args.seed)
    s_y_km = clustering_model.fit_predict(test_s_embeddings_np)
    distances = np.linalg.norm(test_s_embeddings_np[None, :, :] - clustering_model.cluster_centers_[:, None, :], axis=-1)
    s_cluster_ids = np.argsort(distances, axis=1)[:, :5]

    clustering_model = clustering_method(n_clusters=args.num_clusters, random_state=args.seed)
    pi_y_km = clustering_model.fit_predict(pi_skill_embeddings)
    distances = np.linalg.norm(pi_skill_embeddings[None, :, :] - clustering_model.cluster_centers_[:, None, :], axis=-1)
    pi_cluster_ids = np.argsort(distances, axis=1)[:, :5]

    with open(os.path.join(save_path, "config.json"), 'w') as f:
        cfg = vars(args)
        cfg.update({
            "NMI_GT_QS_skill": {
                "Q_skill": normalized_mutual_info_score(qs_skill_y_km, y_km),
                "pi_skill": normalized_mutual_info_score(qs_skill_y_km, pi_y_km),
                "Q": normalized_mutual_info_score(qs_skill_y_km, q_y_km),
                "QS_skill": normalized_mutual_info_score(qs_skill_y_km, qs_skill_y_km),
                "QS": normalized_mutual_info_score(qs_skill_y_km, qs_y_km),
            },
            "NMI_GT_QS": {
                "Q_skill": normalized_mutual_info_score(qs_y_km, y_km),
                "pi_skill": normalized_mutual_info_score(qs_y_km, pi_y_km),
                "Q": normalized_mutual_info_score(qs_y_km, q_y_km),
                "QS_skill": normalized_mutual_info_score(qs_y_km, qs_skill_y_km),
                "QS": normalized_mutual_info_score(qs_y_km, qs_y_km),
            },
            "NMI_GT_S": {
                "Q_skill": normalized_mutual_info_score(s_y_km, y_km),
                "pi_skill": normalized_mutual_info_score(s_y_km, pi_y_km),
                "Q": normalized_mutual_info_score(s_y_km, q_y_km),
                "QS_skill": normalized_mutual_info_score(s_y_km, qs_skill_y_km),
                "QS": normalized_mutual_info_score(s_y_km, qs_y_km),
            }
        })
        json.dump(cfg, f, indent=2, separators=(',', ': '))
        print("Save the model and config to %s" %save_path)

    fig, axes = plt.subplots(6, 6, figsize=(24, 24))
    for i, (plot_embedding, name, cluster_ids) in enumerate(zip(
        [q_skill_embeddings, pi_skill_embeddings, test_q_embeddings_np, qs_skill_embeddings, test_qs_embeddings_np, test_s_embeddings_np],
        ["Q_Skill", "Pi_skill", "Q", "QS_Skill", "QS", "S"], [q_skill_cluster_ids, pi_cluster_ids, q_cluster_ids, qs_skill_cluster_ids, qs_cluster_ids, s_cluster_ids]
    )):
        plot_algo = TSNE if args.plot_model == "tsne" else PCA
        pca_model = plot_algo(n_components=2, random_state=args.seed)
        transformed = pca_model.fit_transform(plot_embedding)

        axes[i][0].scatter(x=transformed[:, 0], y=transformed[:, 1], c=y_km, s=50, cmap=plt.cm.Paired, alpha=0.1)
        axes[i][0].set_xticks([])
        axes[i][0].set_yticks([])
        axes[i][0].set_title(f"plot ({name}), cluster (Q_skill)")

        axes[i][1].scatter(x=transformed[:, 0], y=transformed[:, 1], c=pi_y_km, s=50, cmap=plt.cm.Paired, alpha=0.1)
        axes[i][1].set_xticks([])
        axes[i][1].set_yticks([])
        axes[i][1].set_title(f"plot ({name}), cluster (Pi_skill)")

        axes[i][2].scatter(x=transformed[:, 0], y=transformed[:, 1], c=q_y_km, s=50, cmap=plt.cm.Paired, alpha=0.1)
        axes[i][2].set_xticks([])
        axes[i][2].set_yticks([])
        axes[i][2].set_title(f"plot ({name}), cluster (Q)")

        axes[i][3].scatter(x=transformed[:, 0], y=transformed[:, 1], c=qs_skill_y_km, s=50, cmap=plt.cm.Paired, alpha=0.1)
        axes[i][3].set_xticks([])
        axes[i][3].set_yticks([])
        axes[i][3].set_title(f"plot ({name}), cluster (QS_skill)")

        axes[i][4].scatter(x=transformed[:, 0], y=transformed[:, 1], c=qs_y_km, s=50, cmap=plt.cm.Paired, alpha=0.1)
        axes[i][4].set_xticks([])
        axes[i][4].set_yticks([])
        axes[i][4].set_title(f"plot ({name}), cluster (QS)")

        axes[i][5].scatter(x=transformed[:, 0], y=transformed[:, 1], c=s_y_km, s=50, cmap=plt.cm.Paired, alpha=0.1)
        axes[i][5].set_xticks([])
        axes[i][5].set_yticks([])
        axes[i][5].set_title(f"plot ({name}), cluster (S)")

        cluster_samples = {i: [{"Q": q_text[cid], "A": s_text[cid]} for cid in cluster_ids[i]] for i in range(len(cluster_ids))}
        json.dump(cluster_samples, open(os.path.join(save_path, f"{args.test_split}-{args.num_clusters}-{name}-cluster.json"), 'w'), indent=2, ensure_ascii=False)
    
    # import ipdb; ipdb.set_trace()
    # plt.tight_layout()
    # fig.legend(list(range(args.num_clusters)), loc='lower center', ncol=args.num_clusters, fontsize=20)


    # plt.title(f"{args.num_clusters} clustering on {args.split}")
    plt.savefig(os.path.join(save_path, f"{args.plot_model}-{args.test_split}-{args.num_clusters}-clustering.png"), dpi=200)
    plt.close()

    # Plots used in the paper
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # plt.rcParams['text.usetex'] = True
    colormap = plt.cm.nipy_spectral
    colors = colormap(np.linspace(0, 1, args.num_clusters))
    markers = ["o", "v", "x"]

    pca_model = plot_algo(n_components=2, random_state=args.seed)
    transformed = pca_model.fit_transform(qs_skill_embeddings)

    if ground_truth_cluster_ids is not None:
        qs_skill_y_km = ground_truth_cluster_ids
    axes[0][0].set_prop_cycle('color', colors)
    for i in np.unique(qs_skill_y_km):
        axes[0][0].scatter(x=transformed[qs_skill_y_km == i, 0], y=transformed[qs_skill_y_km == i, 1], label=i, alpha=0.7, marker=markers[i % 3], s=20)

    axes[0][0].set_xticks([]) 
    axes[0][0].set_yticks([])
    axes[0][0].set_title('Reasoning skill of (Q, R)', fontsize='x-large')

    pca_model = plot_algo(n_components=2, random_state=args.seed)
    transformed = pca_model.fit_transform(q_skill_embeddings)
    axes[0][1].set_prop_cycle('color', colors)
    for i in np.unique(qs_skill_y_km):
        axes[0][1].scatter(x=transformed[qs_skill_y_km == i, 0], y=transformed[qs_skill_y_km == i, 1], label=i, alpha=0.7, marker=markers[i % 3], s=20)
    axes[0][1].set_xticks([])
    axes[0][1].set_yticks([])
    axes[0][1].set_title("Reasoning skill of Q", fontsize='x-large')

    pca_model = plot_algo(n_components=2, random_state=args.seed)
    transformed = pca_model.fit_transform(test_q_embeddings_np)
    axes[1][0].set_prop_cycle('color', colors)
    for i in np.unique(qs_skill_y_km):
        axes[1][0].scatter(x=transformed[qs_skill_y_km == i, 0], y=transformed[qs_skill_y_km == i, 1], label=i, alpha=0.7, marker=markers[i % 3], s=20)
    axes[1][0].set_xticks([])
    axes[1][0].set_yticks([])
    axes[1][0].set_title("Raw question embedding", fontsize='x-large')

    pca_model = plot_algo(n_components=2, random_state=args.seed)
    transformed = pca_model.fit_transform(test_s_embeddings_np)
    axes[1][1].set_prop_cycle('color', colors)
    for i in np.unique(qs_skill_y_km):
        axes[1][1].scatter(x=transformed[qs_skill_y_km == i, 0], y=transformed[qs_skill_y_km == i, 1], label=i, alpha=0.7, marker=markers[i % 3], s=20)
    axes[1][1].set_xticks([])
    axes[1][1].set_yticks([])
    axes[1][1].set_title("Raw rationale embedding", fontsize='x-large')

    handles, labels = axes[1][1].get_legend_handles_labels()
    tabmwp_idx2label = {
        0: "Compute statistics",
        1: "Compute rate of change",
        2: "Compute money cost",
        3: "Filter tree leaves",
        4: "Addtion/subtraction",
        5: "Search minimum/maximum",
        6: "Multiplication",
        7: "Filter table entries",
        8: "Compute probability",
        9: "Shortage or surplus?",
        10: "Reason time schedule",
        11: "Compare numbers",
        12: "Others",
    }
    labels = list(tabmwp_idx2label.values())
    fig.legend(handles, labels, loc='upper left', ncol=1, bbox_to_anchor=(0.9, 0.8), fontsize='large', title="Reasoning skills", title_fontsize='x-large')
    
    plt.savefig(os.path.join(save_path, "paper_fig.pdf"), dpi=300, bbox_inches="tight")
    plt.close()