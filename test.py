from src.models import Generator, Discriminator

# g = Generator((64, 64, 3))
# g.summary()
d = Discriminator((256, 256, 3))
d.summary()