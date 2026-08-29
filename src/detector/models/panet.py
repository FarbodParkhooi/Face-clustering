# panet.py

from modules.config import DetectorConfigs
from torch import nn

cfg = DetectorConfigs()

class PANet(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.cnv2 = nn.Conv2d(in_channels=256, out_channels=cfg.fpn_channels, kernel_size=1, stride=1, padding=0)
		self.cnv3 = nn.Conv2d(in_channels=512, out_channels=cfg.fpn_channels, kernel_size=1, stride=1, padding=0)
		self.cnv4 = nn.Conv2d(in_channels=1024, out_channels=cfg.fpn_channels, kernel_size=1, stride=1, padding=0)
		self.cnv5 = nn.Conv2d(in_channels=2048, out_channels=cfg.fpn_channels, kernel_size=1, stride=1, padding=0)
		
		self.cnv_p5 = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=1, padding=1)
		self.ups_p5 = nn.Upsample(scale_factor=2, mode="nearest")
		
		self.cnv_p4 = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=1, padding=1)
		self.ups_p4 = nn.Upsample(scale_factor=2, mode="nearest")
		
		self.cnv_p3 = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=1, padding=1)
		self.ups_p3 = nn.Upsample(scale_factor=2, mode="nearest")
		
		self.cnv_p2 = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=1, padding=1)
		
		self.cnv_n3 = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=2, padding=1)
		self.cnv_p3_out = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=1, padding=1)
		
		self.cnv_n4 = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=2, padding=1)
		self.cnv_p4_out = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=1, padding=1)
		
		self.cnv_n5 = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=2, padding=1)
		self.cnv_p5_out = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=1, padding=1)
		
		self.cnv_p6_out = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.fpn_channels, kernel_size=3, stride=2, padding=1)
		
	def forward(self, features):
		C2, C3, C4, C5 = features["C2"], features["C3"], features["C4"], features["C5"]
		
		# Top-down path
		lat2, lat3, lat4, lat5 = self.cnv2(C2), self.cnv3(C3), self.cnv4(C4), self.cnv5(C5)
		
		p5 = self.cnv_p5(lat5) 
		p5_ups = self.ups_p5(p5)
		
		lat4 = lat4 + p5_ups
		p4 = self.cnv_p4(lat4)
		p4_ups = self.ups_p4(p4)
		
		lat3 = lat3 + p4_ups
		p3 = self.cnv_p3(lat3)
		p3_ups = self.ups_p3(p3)
		
		lat2 = lat2 + p3_ups
		p2 = self.cnv_p2(lat2)
		
		# Bottom-up path
		n3 = self.cnv_n3(p2)
		p3_out = self.cnv_p3_out(n3+p3)
		
		n4 = self.cnv_n4(p3_out)
		p4_out = self.cnv_p4_out(n4+p4)
		
		n5 = self.cnv_n5(p4_out)
		p5_out = self.cnv_p5_out(n5+p5)
		
		if cfg.use_p6:
			p6_out = self.cnv_p6_out(p5_out)
			
			return {
				"P2" : p2,
				"P3" : p3_out,
				"P4" : p4_out,
				"P5" : p5_out,
				"P6" : p6_out
			}
		
		return {
				"P2" : p2,
				"P3" : p3_out,
				"P4" : p4_out,
				"P5" : p5_out
			}