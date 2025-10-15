from sbi.inference import NPE, NLE, NRE


def NRE_sequential_inference(prior, obs, simulator, n_sims, rounds):
    inference = NRE(prior)
    proposal = prior
    for _ in range(rounds):
        theta = proposal.sample((n_sims,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(
            mcmc_method="slice_np_vectorized",
            mcmc_parameters={"num_chains": 20, "thin": 5},
        )
        proposal = posterior.set_default_x(obs)

    return posterior


def NLE_sequential_inference(prior, obs, simulator, n_sims, rounds):
    inference = NLE(prior)
    proposal = prior

    for _ in range(rounds):
        theta = proposal.sample((n_sims,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(
            mcmc_method="slice_np_vectorized",
            mcmc_parameters={"num_chains": 20, "thin": 5},
        )
        proposal = posterior.set_default_x(obs)

    return posterior


def NPE_sequential_inference(prior, obs, simulator, n_sims, rounds):
    inference = NPE(prior=prior)
    proposal = prior

    for _ in range(rounds):
        theta = proposal.sample((n_sims,))
        x = simulator(
            theta,
        )
        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train(show_train_summary=True)
        posterior = inference.build_posterior(density_estimator).set_default_x(obs)
        proposal = posterior

    return posterior
