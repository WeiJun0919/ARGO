from models.downstream_model import DownstreamModel


def infer_model_type(dataset):
    raw = str(getattr(dataset, "modality", "") or "").lower()
    name = str(getattr(dataset, "name", "") or "").lower()

    if raw in {"tabular", "adult", "synthetic", "csv"} or name in {"adult", "synthetic"}:
        return "tabular"
    if raw in {"text", "imdb"} or name == "imdb":
        return "text"
    if raw in {"image", "cifar10", "cifar"} or name == "cifar10":
        return "image"

    return "auto"


def build_downstream_model_for_dataset(
    dataset,
    random_state=42,
    device="auto",
    image_epochs=30,
    image_batch_size=128,
    image_num_workers=0,
    image_lr=0.003,
    image_backbone="resnet50_pretrained",
):
    model_type = infer_model_type(dataset)
    return (
        DownstreamModel(
            random_state=random_state,
            model_type=model_type,
            device=device,
            image_epochs=image_epochs,
            image_batch_size=image_batch_size,
            image_num_workers=image_num_workers,
            image_lr=image_lr,
            image_backbone=image_backbone,
        ),
        model_type,
    )
