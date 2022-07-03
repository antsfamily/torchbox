    def __init__(self, inchannels, logdet=True):
    def initialize(self, input):
    def forward(self, input):
    def reverse(self, output):
    def __init__(self, inchannels):
    def forward(self, input):
    def reverse(self, output):
    def __init__(self, inchannels):
    def forward(self, input):
    def calc_weight(self):
    def reverse(self, output):
    def __init__(self, inchannels, out_channel, padding=1):
    def forward(self, input):
    def __init__(self, inchannels, filter_size=512, affine=True):
    def forward(self, input):
    def reverse(self, output):
    def __init__(self, inchannels, affine=True, convlu=True):
    def forward(self, input):
    def reverse(self, output):
def gaussian_log_p(x, mean, log_sd):
def gaussian_sample(eps, mean, log_sd):
    def __init__(self, inchannels, nflow, split=True, affine=True, convlu=True):
    def forward(self, input):
    def reverse(self, output, eps=None, reconstruct=False):
    def __init__(
    def forward(self, input):
    def reverse(self, z_list, reconstruct=False):
