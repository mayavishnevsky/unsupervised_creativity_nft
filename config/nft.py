import importlib.util
import os
import ml_collections

_base_path = os.path.join(os.path.dirname(__file__), "base.py")
_base_spec = importlib.util.spec_from_file_location("diffusion_nft_base_config", _base_path)
base = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(base)


def get_config(name):
    return globals()[name]()


def _get_config(base_model="sd3", n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name=""):
    config = base.get_config()
    assert base_model in ["sd3"]
    assert dataset in ["pickscore", "ocr", "geneval"]

    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    if base_model == "sd3":
        config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 4.5
        config.resolution = 512
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7
        bsz = 9

    config.sample.num_image_per_prompt = 24
    num_groups = 48

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    # special design, the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = 14 if dataset == "geneval" else 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval" if dataset == "geneval" else "general_ocr"

    config.run_name = f"nft_{base_model}_{name}"
    config.save_dir = f"logs/nft/{base_model}/{name}"
    config.reward_fn = reward_fn

    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"

    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config


def sd3_ocr():
    reward_fn = {
        "ocr": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=8, gradient_step_per_epoch=2, dataset="ocr", reward_fn=reward_fn, name="ocr"
    )
    config.beta = 0.1
    config.decay_type = 2
    return config


def sd3_geneval():
    reward_fn = {
        "geneval": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="geneval",
    )
    return config


def sd3_pickscore():
    reward_fn = {
        "pickscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore",
    )
    return config


def sd3_hpsv2():
    reward_fn = {
        "hpsv2": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=8, gradient_step_per_epoch=1, dataset="pickscore", reward_fn=reward_fn, name="hpsv2"
    )
    return config


def sd3_multi_reward():
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "clipscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.sample.num_steps = 25
    config.beta = 0.1
    return config


def sd3_iem_same_prompt_partiprompts():
    """NFT counterpart of RAM run 190607, adapted to two GPUs."""

    config = base.get_config()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config.base_model = "sd3"
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.pretrained.local_files_only = True
    config.dataset = os.path.join(repo_root, "dataset/partiprompts_basic")
    config.prompt_fn = "general_ocr"
    config.resolution = 512
    config.seed = 0
    config.mixed_precision = "bf16"
    config.num_epochs = 20
    config.save_freq = 1
    config.eval_freq = 1
    config.eval_before_training = True
    config.run_name = "nft_sd35m_iem_same_prompt_partiprompts_sigma0p009_1000"
    config.save_dir = os.path.join(repo_root, "outputs", config.run_name)
    config.wandb_project = "unsupervised_creativity_nft"
    config.wandb_entity = "mayavishnevsky-tel-aviv-university"
    config.wandb_mode = "online"
    config.baseline_lora_path = (
        "/home/dcor/shellygo/DiffusionNFT/logs/nft/sd3/multi_reward/"
        "checkpoints/checkpoint-12"
    )

    config.sample.num_steps = 25
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    config.sample.noise_level = 0.7
    config.sample.num_image_per_prompt = 24
    config.sample.num_prompt_groups = 48
    config.sample.train_batch_size = 6
    config.sample.num_batches_per_epoch = 96
    config.sample.test_batch_size = 2
    config.sample.global_std = False
    config.sample.ram_aligned_prompt_sampling = False

    config.beta = 0.1
    config.decay_type = 1
    config.train.batch_size = 8
    config.train.gradient_accumulation_steps = 72
    config.train.learning_rate = 3e-4
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.99
    config.train.adam_weight_decay = 1e-4
    config.train.beta = 0.01
    config.train.adv_mode = "all"
    config.train.lora_rank = 32
    config.train.lora_alpha = 64

    config.validation = validation = ml_collections.ConfigDict()
    validation.prompt_files = [
        os.path.join(repo_root, "dataset/partiprompts_basic/test.txt"),
        os.path.join(repo_root, "dataset/partiprompts_imagination/test.txt"),
    ]
    validation.prompt_count = 30
    validation.batch_size = 2
    validation.base_seed = 0
    validation.prompt_seed = validation.base_seed

    config.creativity = creativity = ml_collections.ConfigDict()
    creativity.enabled = True
    creativity.distance_metric = "iem"
    creativity.iem_objective = "expected_squared_distance"
    creativity.distance_metrics = ["iem"]
    creativity.reward_weights = ml_collections.ConfigDict({"iem": 1.0})
    creativity.sigma_min = 0.009
    creativity.sigma_max = 1000.0
    creativity.num_steps = 64
    # Levels per forward; must divide num_steps. Effective batch also includes endpoints.
    creativity.level_batch_size = 4
    creativity.noise_table_count = 8
    creativity.reference_prompt_mode = "same_prompt"
    creativity.reference_samples_per_prompt = 256
    creativity.reference_selection_mode = "all"
    creativity.nearest_reference_fraction = 0.1
    creativity.reference_prompt_files = [
        os.path.join(repo_root, "dataset/partiprompts_basic/train.txt")
    ]
    creativity.candidate_prompt_files = [
        os.path.join(repo_root, "dataset/partiprompts_basic/train.txt")
    ]
    creativity.candidate_prompt_sampling_mode = "independent_epoch"
    creativity.candidate_prompt_source_weights = None
    creativity.reference_cache_dir = None
    creativity.reference_cache_spec_sha256 = None
    creativity.reference_samples_per_epoch = 48
    creativity.reference_bank_size = 4096
    creativity.reference_subset_size = 512
    creativity.reference_batch_size = 2
    creativity.feature_batch_size = 2
    creativity.seed = 0
    creativity.resolution = 512
    creativity.num_inference_steps = 25
    creativity.guidance_scale = 1.0
    creativity.clip_model_id = "openai/clip-vit-large-patch14"
    creativity.dino_model_id = "facebook/dinov2-base"
    creativity.tpips_model_id = "sywang/TPIPS-Embed-Qwen3VL-8B"
    creativity.tpips_batch_size = 1
    return config



def sd3_iem_same_prompt_partiprompts_ram_aligned():
    """Use RAM epoch prompts, validation inputs, and cached references."""

    config = sd3_iem_same_prompt_partiprompts()
    default_cache_dir = (
        "/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/"
        "reference_cache/"
        "sd35m_cfgdistilled_partiprompts_basic_512_n20_seed0_refs256_e20_iem_s0p009_1000_g64_v2"
    )
    config.sample.ram_aligned_prompt_sampling = True
    config.validation.prompt_seed = (
        int(config.validation.base_seed) + 3_000_009
    )
    config.creativity.reference_cache_dir = os.environ.get(
        "RAM_REFERENCE_CACHE_DIR", default_cache_dir
    )
    config.creativity.reference_cache_spec_sha256 = os.environ.get(
        "RAM_REFERENCE_CACHE_SPEC_SHA256",
        "d2ac2f59a948f67532f6f9dfa100992ae08fe6683bbc59768f2cccc38f566570",
    )
    # The shared cloud was generated by RAM with its 20-step Euler sampler.
    config.creativity.num_inference_steps = 20
    config.run_name += "_ram_aligned"
    config.save_dir += "_ram_aligned"
    return config


def sd3_iem_same_prompt_mixed_no_repeat():
    """Use fixed 50/25/25 source quotas with per-source no-repeat cycles."""

    config = sd3_iem_same_prompt_partiprompts()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_files = [
        os.path.join(repo_root, "dataset/hpdv3_plusplus/train.txt"),
        os.path.join(repo_root, "dataset/creative_chat/creative_chat_train.txt"),
        os.path.join(repo_root, "dataset/partiprompts_basic/train.txt"),
    ]
    config.creativity.candidate_prompt_files = prompt_files
    config.creativity.reference_prompt_files = prompt_files
    config.creativity.candidate_prompt_sampling_mode = "no_repeat_cycle"
    config.creativity.candidate_prompt_source_weights = [0.5, 0.25, 0.25]
    config.run_name += "_mixed_no_repeat"
    config.save_dir += "_mixed_no_repeat"
    return config


def sd3_iem_same_prompt_partiprompts_smoke():
    """Two-GPU end-to-end smoke test for the full creativity path."""

    config = sd3_iem_same_prompt_partiprompts()
    config.num_epochs = 1
    config.sample.num_steps = 2
    config.sample.eval_num_steps = 2
    config.sample.num_image_per_prompt = 2
    config.sample.num_prompt_groups = 2
    config.sample.train_batch_size = 2
    config.sample.num_batches_per_epoch = 1
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 1
    config.validation.prompt_count = 2
    config.validation.batch_size = 1
    config.creativity.num_steps = 2
    config.creativity.level_batch_size = 1
    config.creativity.noise_table_count = 2
    config.creativity.reference_samples_per_prompt = 2
    config.creativity.reference_batch_size = 1
    config.creativity.feature_batch_size = 1
    config.creativity.num_inference_steps = 2
    config.run_name += "_smoke"
    config.save_dir += "_smoke"
    return config
