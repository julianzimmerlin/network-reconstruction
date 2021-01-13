import torch
loss_fn = torch.nn.NLLLoss()

def train_dynamics_learner_batch(optimizer, dynamics_learner, matrix, data, device, is_continuous,  optimize=True):
    if optimize:
        optimizer.zero_grad()
    adjs = matrix.repeat(data.size()[0], 1, 1)
    # data shape: (BATCH_SIZE, NUM_NODES, NUM_STEPS, INPUT_SIZE)
    input = data[:, :, 0, :]
    target = data[:, :, 1:, :]
    output = input

    outputs = torch.zeros(data.size()[0], data.size()[1], data.size()[2] - 1, data.size(3), device=device)
    for t in range(data.size()[2] - 1):
        # use previous output to predict next step
        output = dynamics_learner(output, adjs)
        outputs[:, :, t, :] = output
    if is_continuous:
        loss = torch.mean(torch.abs(outputs - target))  # L1 LOSS
    else:
        output = output.permute(0, 2, 1)
        target = target[:, :, 0, 1].long()
        loss = loss_fn(output, target)
    loss.backward()
    if optimize:
        optimizer.step()
    return loss

def train_network_generator_batch(optimizer, dynamics_learner, network_generator, data, device, is_continuous):
    optimizer.zero_grad()
    out_matrix = network_generator.get_matrix()
    adjs = out_matrix.repeat(data.size()[0], 1, 1)
    input = data[:, :, 0, :]
    target = data[:, :, 1:, :]
    output = input

    outputs = torch.zeros(data.size()[0], data.size()[1], data.size()[2] - 1, data.size(3), device=device)
    for t in range(data.size()[2] - 1):
        # use previous output to predict next step
        output = dynamics_learner(output, adjs)
        outputs[:, :, t, :] = output

    if is_continuous:
        loss = torch.mean(torch.abs(outputs - target))  # L1 LOSS
    else:
        output = output.permute(0, 2, 1)
        target = target[:, :, 0, 1].long()
        loss = loss_fn(output, target)
    
    loss.backward()
    optimizer.step()
    return loss

def train_dynamics(dynamics_learner, network_generator, optimizer_dyn, data_loader, num_steps, device, is_continuous, is_first, tracker):
    matrix = network_generator.get_matrix_hard()
    step_losses = list()
    for step in range(num_steps):
        batch_losses = list()
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            loss = train_dynamics_learner_batch(optimizer_dyn, dynamics_learner, matrix,
                                             data, device, is_continuous)
            batch_losses.append(loss)
            if is_first and step==0 and batch_idx==0:
                tracker.track(network_generator.get_matrix_hard().to(torch.float32), loss=loss.data.item())
        step_loss = torch.stack(batch_losses).mean()
        step_losses.append(step_loss)
        print('Mean Loss in Dyn Epoch ' + str(step) + ': ' + str(step_loss.item()))
    return torch.mean(torch.stack(step_losses)).item()

def train_network(dynamics_learner, network_generator, optimizer_dyn, data_loader, num_steps, device, is_continuous):
    step_losses = list()
    for step in range(num_steps):
        batch_losses = list()
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            for p in dynamics_learner.parameters():
                p.requires_grad = False
            loss = train_network_generator_batch(optimizer_dyn, dynamics_learner, network_generator,
                                             data, device, is_continuous)
            for p in dynamics_learner.parameters():
                p.requires_grad = True
            batch_losses.append(loss)
        step_loss = torch.stack(batch_losses).mean()
        step_losses.append(step_loss)
        print('Mean Loss in Net Epoch ' + str(step) + ': ' + str(step_loss.item()))
    return torch.mean(torch.stack(step_losses)).item()