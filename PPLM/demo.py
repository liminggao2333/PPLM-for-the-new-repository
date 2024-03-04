
from run_pplm import run_pplm_example
run_pplm_example(
    cond_text="The potato",
    num_samples=3,
    bag_of_words='military',
    length=50,
    stepsize=0.03,
    sample=True,
    num_iterations=3,
    window_length=5,
    gamma=1.5,
    gm_scale=0.95,
    kl_scale=0.01,
    verbosity='regular'
)
# run_pplm_example(
#     cond_text="My dog died",
#     num_samples=3,
#     discrim='sentiment',
#     class_label='very_positive',
#     length=50,
#     stepsize=0.05,
#     sample=True,
#     num_iterations=10,
#     gamma=1,
#     gm_scale=0.9,
#     kl_scale=0.02,
#     verbosity='regular'
# )