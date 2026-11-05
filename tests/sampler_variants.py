"""Native-container wrappers that preserve positional forwarding."""


def wrap_sample(original, variant):
    if variant == 'optional':
        def sample(self, model_wrap, sigmas, extra_args, callback, noise,
                   latent_image=None, denoise_mask=None, disable_pbar=False,
                   progress_hook=None):
            return original(self, model_wrap, sigmas, extra_args, callback, noise,
                            latent_image, denoise_mask, disable_pbar)
    elif variant == 'renamed':
        def sample(owner, guider, schedule, options, observer, initial_noise,
                   latent=None, mask=None, quiet=False):
            return original(owner, guider, schedule, options, observer, initial_noise,
                            latent, mask, quiet)
    elif variant == 'variadic':
        def sample(self, *args, **kwargs):
            return original(self, *args, **kwargs)
    else:
        raise ValueError(variant)
    return sample
