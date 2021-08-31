import numpy as np
import tensorflow as tf
import timeit



Z_DIM = 128

#HYPERPARAMETERS
SCALE = 1
BATCH_SIZE_DISC = 64
BATCH_SIZE_GEN = 64
BATCH_SIZE_MAX = max(BATCH_SIZE_DISC, BATCH_SIZE_GEN)
KAPPA = 0.5
SOFT_WEIGHT_THRESHOLD= 0.5



class ContinuousConditionalGAN(tf.keras.Model):
    def __init__(self, discriminator, generator, embedding):
        super(ContinuousConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.embedding = embedding
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]


    def compile(self, disc_optimizer, gen_optimizer, clip_label, threshold_type, loss_type):
        super(ContinuousConditionalGAN, self).compile()
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.clip_label = clip_label
        self.threshold_type = threshold_type
        self.loss_type = loss_type


    def train_step(self, data):
        train_images, train_labels = data


        unique_train_labels = np.sort(np.array(list(set(train_labels))))

        batch_target_labels_in_dataset = np.random.choice(unique_train_labels, size=BATCH_SIZE_MAX, replace=True)
        batch_epsilons = np.random.normal(0, SCALE, BATCH_SIZE_MAX)
        batch_target_labels_with_epsilon = batch_target_labels_in_dataset + batch_epsilons

        if self.clip_label:
            batch_target_labels_with_epsilon = np.clip(batch_target_labels_with_epsilon, 0.0, 1.0)


        batch_target_labels = batch_target_labels_with_epsilon[0:BATCH_SIZE_DISC]
        batch_real_indx = np.zeros(BATCH_SIZE_DISC, dtype=int)
        batch_fake_labels = np.zeros(BATCH_SIZE_DISC)


        #Obtain discriminator batches
        for i in range(BATCH_SIZE_DISC):
            if self.threshold_type == 'hard':
                indx_real_in_vicinity = np.where(np.abs(train_labels - batch_target_labels[i]) <= KAPPA)[0]
            else:

                indx_real_in_vicinity = np.where((train_labels - batch_target_labels[i])**2 <= - np.log(SOFT_WEIGHT_THRESHOLD / KAPPA))[0]

            while len(indx_real_in_vicinity) < 1:
                batch_epsilons_i = np.random.normal(0, SCALE, 1)
                batch_target_labels[i] = batch_target_labels_in_dataset[i] + batch_epsilons_i

                if self.clip_label:
                    batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)

                if self.threshold_type == 'hard':
                    indx_real_in_vicinity = np.where(np.abs(train_labels - batch_target_labels[i]) <= KAPPA)[0]
                else:
                    indx_real_in_vicinity = np.where((train_labels - batch_target_labels[i])**2 <= - np.log(SOFT_WEIGHT_THRESHOLD) / KAPPA)[0]

            assert len(indx_real_in_vicinity) >= 1

            batch_real_indx[i] = np.random.choice(indx_real_in_vicinity, size=1)[0]


            if self.threshold_type == 'hard':
                lower_bound = batch_target_labels[i] - KAPPA
                upper_bound = batch_target_labels[i] + KAPPA
            else:
                lower_bound = batch_target_labels[i] - np.sqrt( - np.log(SOFT_WEIGHT_THRESHOLD) / KAPPA)
                upper_bound = batch_target_labels[i] + np.sqrt( - np.log(SOFT_WEIGHT_THRESHOLD) / KAPPA)

            lower_bound = max(0.0, lower_bound)
            upper_bound = min(upper_bound, 1.0)

            assert lower_bound <= upper_bound
            assert lower_bound >= 0 and upper_bound >= 0
            assert lower_bound <= 1 and upper_bound <= 1


            batch_fake_labels[i] = np.random.uniform(lower_bound, upper_bound, size=1)[0]


        #Train the discriminator
        batch_real_images = [train_images[index] for index in batch_real_indx]
        batch_real_images = tf.convert_to_tensor(batch_real_images)

        batch_real_labels = train_labels[batch_real_indx]
        batch_real_labels = tf.convert_to_tensor(batch_real_labels)
        batch_real_labels = tf.reshape(batch_real_labels, [len(batch_real_labels), 1])

        batch_fake_labels = tf.convert_to_tensor(batch_fake_labels)
        batch_fake_labels = tf.reshape(batch_fake_labels, [len(batch_fake_labels), 1])

        batch_target_labels = tf.convert_to_tensor(batch_target_labels)
        batch_target_labels = tf.reshape(batch_target_labels, [len(batch_target_labels), 1])

        z_tensor = tf.random.uniform([BATCH_SIZE_DISC, Z_DIM])

        batch_fake_images = generator((z_tensor, self.embedding(batch_fake_labels)))

        if self.threshold_type == 'soft':
            real_weights = tf.math.exp( - KAPPA*(batch_real_labels - batch_target_labels)**2)
            fake_weights = tf.math.exp( - KAPPA*(batch_fake_labels - batch_target_labels)**2)
        else:
            real_weights = tf.ones(BATCH_SIZE_DISC)
            fake_weights = tf.ones(BATCH_SIZE_DISC)


        with tf.GradientTape() as tape:
            real_dis_out = discriminator(batch_real_images, embedding(batch_target_labels))
            fake_dis_out = discriminator(batch_fake_images, embedding(batch_fake_labels))

            if self.loss_type == 'vanilla':
                real_dis_out = tf.keras.activations.sigmoid(real_dis_out)
                fake_dis_out = tf.keras.activations.sigmoid(fake_dis_out)
                d_loss_real = tf.reshape(-tf.math.log(real_dis_out+1e-20), [-1])
                d_loss_fake = tf.reshape(-tf.math.log(1-fake_dis_out+1e-20), [-1])
            elif self.loss_type == 'hinge':
                d_loss_real = tf.reshape(tf.keras.activations.relu(1.0 - real_dis_out), [-1])
                d_loss_fake = tf.reshape(tf.keras.activations.relu(1.0 + fake_dis_out), [-1])

            d_loss = tf.math.reduce_mean(real_weights * d_loss_real) + tf.math.reduce_mean(fake_weights * d_loss_fake)

            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))



        #Train the generator
        batch_target_labels = batch_target_labels_with_epsilon[0:BATCH_SIZE_GEN]
        batch_target_labels = tf.convert_to_tensor(batch_target_labels)
        batch_target_labels = tf.reshape(batch_target_labels, [len(batch_target_labels), 1])

        z_tensor = tf.random.uniform([BATCH_SIZE_GEN, Z_DIM])

        with tf.GradientTape() as tape:
            batch_fake_images = generator((z_tensor, embedding(batch_target_labels)))
            dis_out = discriminator(batch_fake_images, embedding(batch_target_labels))

            if self.loss_type == 'vanilla':
                dis_out = tf.keras.activations.sigmoid(dis_out)
                g_loss = -tf.math.reduce_mean(tf.math.log(dis_out+1e-20))
            elif self.loss_type == 'hinge':
                g_loss = tf.math.reduce_mean(dis_out)

            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))


        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }






if __name__ == '__main__':

    from utils import read_sequences, read_energy, obtain_csv_names
    from preprocessing import aa_to_tensor, pad_tensor_variable, minmax_label_normalize
    from generator import Generator
    from discriminator import Discriminator
    from embedding import LabelEmbedding

    files = obtain_csv_names()
    sequences = read_sequences(files[0])

    train_images = []
    for sequence in sequences:
        train_images.append(aa_to_tensor(sequence))

    train_images = [pad_tensor_variable(tensor) for tensor in train_images]
    train_images = [tf.reshape(train_images[i], [64,64,1]) for i in range(len(train_images))]

    train_labels = np.abs(read_energy(files[0]))
    train_labels = minmax_label_normalize(train_labels)

    dataset = (train_images, train_labels)

    generator = Generator()
    discriminator = Discriminator()
    embedding = LabelEmbedding(dim_embed=128)


    disc_optimizer = tf.keras.optimizers.Adam(beta_1=0.5)
    gen_optimizer = tf.keras.optimizers.Adam(beta_1=0.5)


    ContCondGan = ContinuousConditionalGAN(discriminator, generator, embedding)

    ContCondGan.compile(disc_optimizer, gen_optimizer, clip_label=False, threshold_type='hard', loss_type='vanilla')



    ContCondGan.train_step(dataset)
