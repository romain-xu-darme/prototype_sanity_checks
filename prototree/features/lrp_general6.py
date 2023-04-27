"""
Taken from https://github.com/SrishtiGautam/PRP
Code built upon LRP code from https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet
"""
import torch
import torch.nn as nn

import copy
import torch.nn.functional as F


#######################################################
# wrappers for autograd type modules
#######################################################

class zeroparam_wrapper_class(nn.Module):
	def __init__(self, module, autogradfunction):
		super(zeroparam_wrapper_class, self).__init__()
		self.module = module
		self.wrapper = autogradfunction

	def forward(self, x):
		y = self.wrapper.apply(x, self.module)
		return y


class oneparam_wrapper_class(nn.Module):
	def __init__(self, module, autogradfunction, parameter1):
		super(oneparam_wrapper_class, self).__init__()
		self.module = module
		self.wrapper = autogradfunction
		self.parameter1 = parameter1

	def forward(self, x):
		y = self.wrapper.apply(x, self.module, self.parameter1)
		return y


class conv2d_zbeta_wrapper_class(nn.Module):
	def __init__(self, module, lrpignorebias, lowest=None, highest=None):
		super(conv2d_zbeta_wrapper_class, self).__init__()

		if lowest is None:
			lowest = torch.min(torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]))
		if highest is None:
			highest = torch.max(torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]))
		assert (isinstance(module, nn.Conv2d))

		self.module = module
		self.wrapper = conv2d_zbeta_wrapper_fct()
		self.lrpignorebias = lrpignorebias
		self.lowest = lowest
		self.highest = highest

	def forward(self, x):
		y = self.wrapper.apply(x, self.module, self.lrpignorebias, self.lowest, self.highest)
		return y


class lrplookupnotfounderror(Exception):
	pass


def get_lrpwrapperformodule(module, lrp_params, lrp_layer2method, thisis_inputconv_andiwant_zbeta=False):
	if isinstance(module, nn.ReLU):
		key = 'nn.ReLU'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return zeroparam_wrapper_class(module, autogradfunction=autogradfunction)

	elif isinstance(module, nn.Sigmoid):
		key = 'nn.Sigmoid'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return zeroparam_wrapper_class(module, autogradfunction=autogradfunction)

	elif isinstance(module, nn.BatchNorm2d):
		key = 'nn.BatchNorm2d'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return zeroparam_wrapper_class(module, autogradfunction=autogradfunction)

	elif isinstance(module, nn.Linear):
		key = 'nn.Linear'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return oneparam_wrapper_class(module, autogradfunction=autogradfunction, parameter1=lrp_params['linear_eps'])

	elif isinstance(module, nn.Conv2d):
		if thisis_inputconv_andiwant_zbeta:
			return conv2d_zbeta_wrapper_class(module, lrp_params['conv2d_ignorebias'])
		else:
			key = 'nn.Conv2d'
			if key not in lrp_layer2method:
				raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
			autogradfunction = lrp_layer2method[key]()
			return oneparam_wrapper_class(module, autogradfunction=autogradfunction,
										  parameter1=lrp_params['conv2d_ignorebias'])

	elif isinstance(module, nn.AdaptiveAvgPool2d):
		key = 'nn.AdaptiveAvgPool2d'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return oneparam_wrapper_class(module, autogradfunction=autogradfunction, parameter1=lrp_params['pooling_eps'])

	elif isinstance(module, nn.AvgPool2d):
		key = 'nn.AvgPool2d'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return oneparam_wrapper_class(module, autogradfunction=autogradfunction, parameter1=lrp_params['pooling_eps'])

	elif isinstance(module, nn.MaxPool2d):
		key = 'nn.MaxPool2d'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return zeroparam_wrapper_class(module, autogradfunction=autogradfunction)

	elif isinstance(module, sum_stacked2):  # resnet specific
		key = 'sum_stacked2'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return oneparam_wrapper_class(module, autogradfunction=autogradfunction, parameter1=lrp_params['eltwise_eps'])

	elif isinstance(module, clamplayer):  # densenet specific
		key = 'clamplayer'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return zeroparam_wrapper_class(module, autogradfunction=autogradfunction)

	elif isinstance(module, tensorbiased_linearlayer):  # densenet specific
		key = 'tensorbiased_linearlayer'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return oneparam_wrapper_class(module, autogradfunction=autogradfunction, parameter1=lrp_params['linear_eps'])

	elif isinstance(module, tensorbiased_convlayer):  # densenet specific
		key = 'tensorbiased_convlayer'
		if key not in lrp_layer2method:
			raise lrplookupnotfounderror("found no dictionary entry in lrp_layer2method for this module name:", key)
		autogradfunction = lrp_layer2method[key]()
		return oneparam_wrapper_class(module, autogradfunction=autogradfunction,
									  parameter1=lrp_params['conv2d_ignorebias'])

	else:
		raise ModuleNotFoundError(f"Unsupported module {type(module)}")

#######################################################
# canonization functions
#######################################################
def resetbn(bn):
	assert (isinstance(bn, nn.BatchNorm2d))
	bnc = copy.deepcopy(bn)
	bnc.reset_parameters()
	return bnc

def bnafterconv_overwrite_intoconv(conv, bn):
	assert (isinstance(bn, nn.BatchNorm2d))
	assert (isinstance(conv, nn.Conv2d))
	s = (bn.running_var + bn.eps) ** .5
	w = bn.weight
	b = bn.bias
	m = bn.running_mean
	conv.weight = torch.nn.Parameter(conv.weight * (w / s).reshape(-1, 1, 1, 1))
	if conv.bias is None:
		conv.bias = torch.nn.Parameter((0 - m) * (w / s) + b)
	else:
		conv.bias = torch.nn.Parameter((conv.bias - m) * (w / s) + b)
	return conv


def getclamplayer(bn):
	assert (isinstance(bn, nn.BatchNorm2d))
	var_bn = (bn.running_var + bn.eps) ** .5
	w_bn = bn.weight
	bias_bn = bn.bias
	mu_bn = bn.running_mean
	if torch.norm(w_bn) > 0:
		thresh = -bias_bn * var_bn / w_bn + mu_bn
		clamplay = clamplayer(thresh, torch.sign(w_bn), forconv=True)
	else:
		raise ValueError(
			'Bad case, not (torch.norm(w_bn) > 0), exiting, see lrp_general*.py, you can outcomment the exit(), '
			'but it means that your batchnorm layer is messed up.')
	return clamplay

def convafterbn_returntensorbiasedconv(conv, bn):
	assert (isinstance(bn, nn.BatchNorm2d))
	assert (isinstance(conv, nn.Conv2d))
	var_bn = (bn.running_var.clone().detach() + bn.eps) ** .5
	w_bn = bn.weight.clone().detach()
	bias_bn = bn.bias.clone().detach()
	mu_bn = bn.running_mean.clone().detach()
	newweight = conv.weight.clone().detach() * (w_bn / var_bn).reshape(1, conv.weight.shape[1], 1, 1)
	inputfornewbias = - (w_bn / var_bn * mu_bn) + bias_bn  # size [nchannels]
	if conv.padding == 0:
		ksize = (conv.weight.shape[2], conv.weight.shape[3])
		inputfornewbias2 = inputfornewbias.unsqueeze(1).unsqueeze(2).repeat((1, ksize[0], ksize[1])).unsqueeze(0)

		with torch.no_grad():
			prebias = conv(inputfornewbias2)
		mi = ((prebias.shape[2] - 1) // 2, (prebias.shape[3] - 1) // 2)
		prebias = prebias.clone().detach()
		newconv_bias = prebias[0, :, mi[0], mi[1]]

		conv2 = copy.deepcopy(conv)
		conv2.weight = torch.nn.Parameter(newweight)
		conv2.bias = torch.nn.Parameter(newconv_bias)
		return conv2
	else:
		spatiallybiasedconv = tensorbiased_convlayer(newweight, conv, inputfornewbias.clone().detach())
		return spatiallybiasedconv


def linearafterbn_returntensorbiasedlinearlayer(linearlayer, bn):
	assert (isinstance(bn, nn.BatchNorm2d))
	assert (isinstance(linearlayer, nn.Linear))
	var_bn = (bn.running_var + bn.eps) ** .5
	w_bn = bn.weight
	bias_bn = bn.bias
	mu_bn = bn.running_mean
	newweight = torch.nn.Parameter(linearlayer.weight.clone().detach() * (w_bn / var_bn))
	inputfornewbias = - (w_bn / var_bn * mu_bn) + bias_bn  # size [nchannels]
	inputfornewbias = inputfornewbias.detach()
	with torch.no_grad():
		newbias = linearlayer.forward(inputfornewbias)
	tensorbias_linearlayer = tensorbiased_linearlayer(linearlayer.in_features, linearlayer.out_features, newweight,
													  newbias.data)
	return tensorbias_linearlayer

###########################################################
# resnet stuff
###########################################################

class eltwisesum2(nn.Module):
	def __init__(self):
		super(eltwisesum2, self).__init__()

	def forward(self, x1, x2):
		return x1 + x2

###########################################################
# densenet stuff
###########################################################

class clamplayer(nn.Module):

	def __init__(self, thresh, w_bn_sign, forconv):
		super(clamplayer, self).__init__()
		if forconv:
			self.thresh = thresh.reshape((-1, 1, 1))
			self.w_bn_sign = w_bn_sign.reshape((-1, 1, 1))
		else:
			self.thresh = thresh
			self.w_bn_sign = w_bn_sign

	def forward(self, x):
		# for channels c with w_bn > 0  -- as checked by (self.w_bn_sign>0)
		# return (x- self.thresh ) * (x>self.thresh) +  self.thresh
		#
		# for channels c with w_bn < 0
		# return thresh if (x>=self.thresh), x  if (x < self. thresh)
		# return (x- self.thresh ) * (x<self.thresh) +  self.thresh
		return (x - self.thresh) * (
				(x > self.thresh) * (self.w_bn_sign > 0) + (x < self.thresh) * (self.w_bn_sign < 0)) + self.thresh


class tensorbiased_linearlayer(nn.Module):
	def __init__(self, in_features, out_features, newweight, newbias):
		super(tensorbiased_linearlayer, self).__init__()
		assert (newbias.numel() == out_features)
		self.linearbase = nn.Linear(in_features, out_features, bias=False)
		self.linearbase.weight = torch.nn.Parameter(newweight)
		self.biastensor = torch.nn.Parameter(newbias)
		self.in_features = in_features
		self.out_features = out_features

	def forward(self, x):
		y = self.linearbase.forward(x) + self.biastensor
		return y


class tensorbiased_convlayer(nn.Module):

	def _clone_module(self, module):
		clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
						  **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
		return clone.to(module.weight.device)

	def __init__(self, newweight, baseconv, inputfornewbias):
		super(tensorbiased_convlayer, self).__init__()

		self.baseconv = baseconv
		self.inputfornewbias = inputfornewbias
		self.conv = self._clone_module(baseconv)
		self.conv.weight = torch.nn.Parameter(newweight)
		self.conv.bias = None
		self.biasmode = 'neutr'

	def gettensorbias(self, x):
		with torch.no_grad():
			tensorbias = self.baseconv(
				self.inputfornewbias.unsqueeze(1).unsqueeze(2).repeat((1, x.shape[2], x.shape[3])).unsqueeze(0))
		return tensorbias

	def forward(self, x):
		if len(x.shape) != 4:
			raise ValueError('bad tensor length')
		if self.inputfornewbias is None:
			return self.conv.forward(x)  # z
		else:
			b = self.gettensorbias(x)
			if self.biasmode == 'neutr':
				# z+=b
				return self.conv.forward(x) + b
			elif self.biasmode == 'pos':
				# z+= torch.clamp(b,min=0)
				return self.conv.forward(x) + torch.clamp(b, min=0)  # z
			elif self.biasmode == 'neg':
				# z+= torch.clamp(b,max=0)
				return self.conv.forward(x) + torch.clamp(b, max=0)  # z


#######################################################
# autograd type modules
#######################################################
class conv2d_beta0_wrapper_fct(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, module, lrpignorebias):
		def configvalues_totensorlist(module):
			propertynames = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
			values = []
			for attr in propertynames:
				v = getattr(module, attr)
				# convert it into tensor
				# has no treatment for booleans yet
				if isinstance(v, int):
					v = torch.tensor([v], dtype=torch.int32, device=module.weight.device)
				elif isinstance(v, tuple):
					# FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
					v = torch.tensor(v, dtype=torch.int32, device=module.weight.device)
				else:
					raise ValueError('Unsupported attribute', attr, type(v))
				values.append(v)
			return propertynames, values

		# stash module config params and trainable params
		propertynames, values = configvalues_totensorlist(module)
		if module.bias is None:
			bias = None
		else:
			bias = module.bias.data.clone()
		lrpignorebiastensor = torch.tensor([lrpignorebias], dtype=torch.bool, device=module.weight.device)
		ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor,
							  *values)  # *values unpacks the list
		return module.forward(x)

	@staticmethod
	def backward(ctx, grad_output):
		input_, conv2dweight, conv2dbias, lrpignorebiastensor, *values = ctx.saved_tensors

		# reconstruct dictionary of config parameters
		def tensorlist_todict(values):
			propertynames = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
			paramsdict = {}
			for i, n in enumerate(propertynames):
				v = values[i]
				if v.numel == 1:
					paramsdict[n] = v.item()
				else:
					alist = v.tolist()
					if len(alist) == 1:
						paramsdict[n] = alist[0]
					else:
						paramsdict[n] = tuple(alist)
			return paramsdict

		paramsdict = tensorlist_todict(values)
		if conv2dbias is None:
			module = nn.Conv2d(**paramsdict, bias=False)
		else:
			module = nn.Conv2d(**paramsdict, bias=True)
			module.bias = torch.nn.Parameter(conv2dbias)

		module.weight = torch.nn.Parameter(conv2dweight)
		pnconv = posnegconv(module, ignorebias=lrpignorebiastensor.item())

		X = input_.clone().detach().requires_grad_(True)
		R = lrp_backward(_input=X, layer=pnconv, relevance_output=grad_output[0], eps0=1e-12, eps=0)
		return R, None, None


class conv2d_zbeta_wrapper_fct(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, module, lrpignorebias, lowest, highest):
		def configvalues_totensorlist(module):
			propertynames = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
			values = []
			for attr in propertynames:
				v = getattr(module, attr)
				# convert it into tensor
				# has no treatment for booleans yet
				if isinstance(v, int):
					v = torch.tensor([v], dtype=torch.int32, device=module.weight.device)
				elif isinstance(v, tuple):
					# FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
					v = torch.tensor(v, dtype=torch.int32, device=module.weight.device)
				else:
					raise ValueError('Unsupported attribute', attr, type(v))
				values.append(v)
			return propertynames, values

		# stash module config params and trainable params
		propertynames, values = configvalues_totensorlist(module)
		if module.bias is None:
			bias = None
		else:
			bias = module.bias.data.clone()
		lrpignorebiastensor = torch.tensor([lrpignorebias], dtype=torch.bool, device=module.weight.device)
		ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, lowest.to(module.weight.device),
							  highest.to(module.weight.device), *values)  # *values unpacks the list
		return module.forward(x)

	@staticmethod
	def backward(ctx, grad_output):
		input_, conv2dweight, conv2dbias, lrpignorebiastensor, lowest_, highest_, *values = ctx.saved_tensors
		# reconstruct dictionary of config parameters
		def tensorlist_todict(values):
			propertynames = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
			paramsdict = {}
			for i, n in enumerate(propertynames):
				v = values[i]
				if v.numel == 1:
					paramsdict[n] = v.item()
				else:
					alist = v.tolist()
					if len(alist) == 1:
						paramsdict[n] = alist[0]
					else:
						paramsdict[n] = tuple(alist)
			return paramsdict

		paramsdict = tensorlist_todict(values)
		if conv2dbias is None:
			module = nn.Conv2d(**paramsdict, bias=False)
		else:
			module = nn.Conv2d(**paramsdict, bias=True)
			module.bias = torch.nn.Parameter(conv2dbias)
		module.weight = torch.nn.Parameter(conv2dweight)

		any_conv = anysign_conv(module, ignorebias=lrpignorebiastensor.item())
		X = input_.clone().detach().requires_grad_(True)
		L = (lowest_ * torch.ones_like(X)).requires_grad_(True)
		H = (highest_ * torch.ones_like(X)).requires_grad_(True)

		with torch.enable_grad():
			Z = any_conv.forward(mode='justasitis', x=X) - any_conv.forward(mode='pos', x=L) - any_conv.forward(
				mode='neg', x=H)
			S = safe_divide(grad_output[0].clone().detach(), Z.clone().detach(), eps0=1e-6, eps=1e-6)
			Z.backward(S)
			R = (X * X.grad + L * L.grad + H * H.grad).detach()

		return R, None, None, None, None  # for  (x, conv2dclass,lrpignorebias, lowest, highest)


class adaptiveavgpool2d_wrapper_fct(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, module, eps):
		def configvalues_totensorlist(module, device):
			propertynames = ['output_size']
			values = []
			for attr in propertynames:
				v = getattr(module, attr)
				# convert it into tensor
				# has no treatment for booleans yet
				if isinstance(v, int):
					v = torch.tensor([v], dtype=torch.int32, device=device)
				elif isinstance(v, tuple):
					# FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
					v = torch.tensor(v, dtype=torch.int32, device=device)
				else:
					raise ValueError('Unsupported attribute', attr, type(v))
				values.append(v)
			return propertynames, values

		# stash module config params and trainable params
		propertynames, values = configvalues_totensorlist(module, x.device)
		epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)
		ctx.save_for_backward(x, epstensor, *values)  # *values unpacks the list
		return module.forward(x)

	@staticmethod
	def backward(ctx, grad_output):
		input_, epstensor, *values = ctx.saved_tensors

		# reconstruct dictionary of config parameters
		def tensorlist_todict(values):
			propertynames = ['output_size']
			paramsdict = {}
			for i, n in enumerate(propertynames):
				v = values[i]
				if v.numel == 1:
					paramsdict[n] = v.item()
				else:
					alist = v.tolist()
					if len(alist) == 1:
						paramsdict[n] = alist[0]
					else:
						paramsdict[n] = tuple(alist)
			return paramsdict

		paramsdict = tensorlist_todict(values)
		eps = epstensor.item()
		# class instantiation
		layerclass = torch.nn.AdaptiveAvgPool2d(**paramsdict)
		X = input_.clone().detach().requires_grad_(True)
		R = lrp_backward(_input=X, layer=layerclass, relevance_output=grad_output[0], eps0=eps, eps=eps)
		return R, None, None


class maxpool2d_wrapper_fct(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, module):
		def configvalues_totensorlist(module, device):
			propertynames = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']
			values = []
			for attr in propertynames:
				v = getattr(module, attr)
				# convert it into tensor
				if isinstance(v, bool):
					v = torch.tensor([v], dtype=torch.bool, device=device)
				elif isinstance(v, int):
					v = torch.tensor([v], dtype=torch.int32, device=device)
				elif isinstance(v, bool):
					v = torch.tensor([v], dtype=torch.int32, device=device)
				elif isinstance(v, tuple):
					# FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
					v = torch.tensor(v, dtype=torch.int32, device=device)
				else:
					raise ValueError('Unsupported attribute', attr, type(v))
				values.append(v)
			return propertynames, values

		# stash module config params and trainable params
		propertynames, values = configvalues_totensorlist(module, x.device)
		ctx.save_for_backward(x, *values)  # *values unpacks the list
		return module.forward(x)

	@staticmethod
	def backward(ctx, grad_output):
		input_, *values = ctx.saved_tensors

		# reconstruct dictionary of config parameters
		def tensorlist_todict(values):
			propertynames = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']
			paramsdict = {}
			for i, n in enumerate(propertynames):
				v = values[i]
				if v.numel == 1:
					paramsdict[n] = v.item()
				else:
					alist = v.tolist()
					if len(alist) == 1:
						paramsdict[n] = alist[0]
					else:
						paramsdict[n] = tuple(alist)
			return paramsdict

		paramsdict = tensorlist_todict(values)
		layerclass = torch.nn.MaxPool2d(**paramsdict)
		X = input_.clone().detach().requires_grad_(True)
		with torch.enable_grad():
			Z = layerclass.forward(X)
		relevance_output_data = grad_output[0].clone().detach().unsqueeze(0)
		Z.backward(relevance_output_data)
		R = X.grad
		return R, None


class avgpool2d_wrapper_fct(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, module, eps):
		def configvalues_totensorlist(module, device):
			propertynames = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']
			values = []
			for attr in propertynames:
				v = getattr(module, attr)
				# convert it into tensor
				if isinstance(v, bool):
					v = torch.tensor([v], dtype=torch.bool, device=device)
				elif isinstance(v, int):
					v = torch.tensor([v], dtype=torch.int32, device=device)
				elif isinstance(v, tuple):
					################
					# FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
					v = torch.tensor(v, dtype=torch.int32, device=device)
				elif v is None:
					pass
				else:
					raise ValueError('Unsupported attribute', attr, type(v))
				values.append(v)
			return propertynames, values

		# stash module config params and trainable params
		propertynames, values = configvalues_totensorlist(module, x.device)
		epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)
		ctx.save_for_backward(x, epstensor, *values)  # *values unpacks the list
		return module.forward(x)

	@staticmethod
	def backward(ctx, grad_output):
		input_, epstensor, *values = ctx.saved_tensors

		# reconstruct dictionary of config parameters
		def tensorlist_todict(values):
			propertynames = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']
			paramsdict = {}
			for i, n in enumerate(propertynames):
				v = values[i]
				if v is None:
					paramsdict[n] = v
				elif v.numel == 1:
					paramsdict[n] = v.item()
				else:
					alist = v.tolist()
					if len(alist) == 1:
						paramsdict[n] = alist[0]
					else:
						paramsdict[n] = tuple(alist)
			return paramsdict

		paramsdict = tensorlist_todict(values)

		layerclass = torch.nn.AvgPool2d(**paramsdict)
		eps = epstensor.item()
		X = input_.clone().detach().requires_grad_(True)
		R = lrp_backward(_input=X, layer=layerclass, relevance_output=grad_output[0], eps0=eps, eps=eps)
		return R, None, None


class max_avgpool2d_wrapper_fct(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, module, eps):
		def configvalues_totensorlist(module, device):
			propertynames = ['kernel_size', 'stride', 'padding', 'ceil_mode']
			values = []
			for attr in propertynames:
				v = getattr(module, attr)
				# convert it into tensor
				if isinstance(v, bool):
					v = torch.tensor([v], dtype=torch.bool, device=device)
				elif isinstance(v, int):
					v = torch.tensor([v], dtype=torch.int32, device=device)
				elif isinstance(v, bool):
					v = torch.tensor([v], dtype=torch.int32, device=device)
				elif isinstance(v, tuple):
					# FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
					v = torch.tensor(v, dtype=torch.int32, device=device)
				else:
					raise ValueError('Unsupported attribute', attr, type(v))
				values.append(v)
			return propertynames, values

		# stash module config params and trainable params
		propertynames, values = configvalues_totensorlist(module, x.device)
		epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)
		ctx.save_for_backward(x, epstensor, *values)  # *values unpacks the list
		return module.forward(x)

	@staticmethod
	def backward(ctx, grad_output):
		input_, epstensor, *values = ctx.saved_tensors

		# reconstruct dictionary of config parameters
		def tensorlist_todict(values):
			propertynames = ['kernel_size', 'stride', 'padding', 'ceil_mode']
			paramsdict = {}
			for i, n in enumerate(propertynames):
				v = values[i]
				if v is None:
					paramsdict[n] = v
				elif v.numel == 1:
					paramsdict[n] = v.item()
				else:
					alist = v.tolist()
					if len(alist) == 1:
						paramsdict[n] = alist[0]
					else:
						paramsdict[n] = tuple(alist)
			return paramsdict

		paramsdict = tensorlist_todict(values)
		layerclass = torch.nn.AvgPool2d(**paramsdict)
		eps = epstensor.item()
		X = input_.clone().detach().requires_grad_(True)
		R = lrp_backward(_input=X, layer=layerclass, relevance_output=grad_output[0], eps0=eps, eps=eps)
		return R, None, None


class relu_wrapper_fct(torch.autograd.Function):  # to be used with generic_activation_pool_wrapper_class(module,this)
	@staticmethod
	def forward(ctx, x, module):
		return module.forward(x)

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output, None


class linearlayer_eps_wrapper_fct(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, module, eps):
		def configvalues_totensorlist(module):
			propertynames = ['in_features', 'out_features']
			values = []
			for attr in propertynames:
				v = getattr(module, attr)
				# convert it into tensor
				# has no treatment for booleans yet
				if isinstance(v, int):
					v = torch.tensor([v], dtype=torch.int32, device=module.weight.device)
				elif isinstance(v, tuple):
					# FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
					v = torch.tensor(v, dtype=torch.int32, device=module.weight.device)
				else:
					raise ValueError('Unsupported attribute', attr, type(v))
				values.append(v)
			return propertynames, values

		# stash module config params and trainable params
		propertynames, values = configvalues_totensorlist(module)
		epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)
		if module.bias is None:
			bias = None
		else:
			bias = module.bias.data.clone()
		ctx.save_for_backward(x, module.weight.data.clone(), bias, epstensor, *values)  # *values unpacks the list
		return module.forward(x)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		input_, weight, bias, epstensor, *values = ctx.saved_tensors

		# reconstruct dictionary of config parameters
		def tensorlist_todict(values):
			propertynames = ['in_features', 'out_features']
			paramsdict = {}
			for i, n in enumerate(propertynames):
				v = values[i]
				if v.numel == 1:
					paramsdict[n] = v.item()
				else:
					alist = v.tolist()
					if len(alist) == 1:
						paramsdict[n] = alist[0]
					else:
						paramsdict[n] = tuple(alist)
			return paramsdict

		paramsdict = tensorlist_todict(values)
		if bias is None:
			module = nn.Linear(**paramsdict, bias=False)
		else:
			module = nn.Linear(**paramsdict, bias=True)
			module.bias = torch.nn.Parameter(bias)

		module.weight = torch.nn.Parameter(weight)
		eps = epstensor.item()
		X = input_.clone().detach().requires_grad_(True)
		R = lrp_backward(_input=X, layer=module, relevance_output=grad_output[0], eps0=eps, eps=eps)
		return R, None, None


class sum_stacked2(nn.Module):
	def __init__(self):
		super(sum_stacked2, self).__init__()

	@staticmethod
	def forward(x):  # from X=torch.stack([X0, X1], dim=0)
		assert (x.shape[0] == 2)
		return torch.sum(x, dim=0)


class eltwisesum_stacked2_eps_wrapper_fct(
	torch.autograd.Function):  # to be used with generic_activation_pool_wrapper_class(module,this)
	@staticmethod
	def forward(ctx, stackedx, module, eps):
		epstensor = torch.tensor([eps], dtype=torch.float32, device=stackedx.device)
		ctx.save_for_backward(stackedx, epstensor)
		return module.forward(stackedx)

	@staticmethod
	def backward(ctx, grad_output):
		stackedx, epstensor = ctx.saved_tensors
		X = stackedx.clone().detach().requires_grad_(True)
		eps = epstensor.item()
		s2 = sum_stacked2().to(X.device)
		Rtmp = lrp_backward(_input=X, layer=s2, relevance_output=grad_output[0], eps0=eps, eps=eps)
		return Rtmp, None, None

#######################################################
# aux input classes
#######################################################
class posnegconv(nn.Module):
	def _clone_module(self, module):
		clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
						  **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
		return clone.to(module.weight.device)

	def __init__(self, conv, ignorebias):
		super(posnegconv, self).__init__()
		self.posconv = self._clone_module(conv)
		self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0)).to(conv.weight.device)
		self.negconv = self._clone_module(conv)
		self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0)).to(conv.weight.device)
		self.anyconv = self._clone_module(conv)
		self.anyconv.weight = torch.nn.Parameter(conv.weight.data.clone()).to(conv.weight.device)

		if ignorebias:
			self.posconv.bias = None
			self.negconv.bias = None
		else:
			if conv.bias is not None:
				self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0))
				self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0))

	def forward(self, x):
		vp = self.posconv(torch.clamp(x, min=0))
		vn = self.negconv(torch.clamp(x, max=0))
		return vp + vn


class anysign_conv(nn.Module):
	def _clone_module(self, module):
		clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
						  **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
		return clone.to(module.weight.device)

	def __init__(self, conv, ignorebias):
		super(anysign_conv, self).__init__()
		self.posconv = self._clone_module(conv)
		self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0)).to(conv.weight.device)
		self.negconv = self._clone_module(conv)
		self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0)).to(conv.weight.device)
		self.jusconv = self._clone_module(conv)
		self.jusconv.weight = torch.nn.Parameter(conv.weight.data.clone()).to(conv.weight.device)

		if ignorebias:
			self.posconv.bias = None
			self.negconv.bias = None
			self.jusconv.bias = None
		else:
			if conv.bias is not None:
				self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0)).to(conv.weight.device)
				self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0)).to(conv.weight.device)
				self.jusconv.bias = torch.nn.Parameter(conv.bias.data.clone()).to(conv.weight.device)

	def forward(self, mode, x):
		if mode == 'pos':
			return self.posconv.forward(x)
		elif mode == 'neg':
			return self.negconv.forward(x)
		elif mode == 'justasitis':
			return self.jusconv.forward(x)
		else:
			raise NotImplementedError("anysign_conv notimpl mode: " + str(mode))

###########################################################
# densenet stuff
###########################################################

class tensorbiased_linearlayer_eps_wrapper_fct(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, module, eps):
		def configvalues_totensorlist(module):
			propertynames = ['in_features', 'out_features']
			values = []
			for attr in propertynames:
				v = getattr(module, attr)
				# convert it into tensor
				# has no treatment for booleans yet
				if isinstance(v, int):
					v = torch.tensor([v], dtype=torch.int32, device=x.device)
				elif isinstance(v, tuple):
					# FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
					v = torch.tensor(v, dtype=torch.int32, device=x.device)
				else:
					raise ValueError('Unsupported attribute', attr, type(v))
				values.append(v)
			return propertynames, values

		# stash module config params and trainable params
		propertynames, values = configvalues_totensorlist(module)
		epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)
		ctx.save_for_backward(x, module.linearbase.weight.data.clone(), module.biastensor.data.clone(), epstensor,
							  *values)
		return module.forward(x)

	@staticmethod
	def backward(ctx, grad_output):
		input_, weight, biastensor, epstensor, *values = ctx.saved_tensors

		# reconstruct dictionary of config parameters
		def tensorlist_todict(values):
			propertynames = ['in_features', 'out_features']
			paramsdict = {}
			for i, n in enumerate(propertynames):
				v = values[i]
				if v.numel == 1:
					paramsdict[n] = v.item()
				else:
					alist = v.tolist()
					if len(alist) == 1:
						paramsdict[n] = alist[0]
					else:
						paramsdict[n] = tuple(alist)
			return paramsdict

		paramsdict = tensorlist_todict(values)
		module = tensorbiased_linearlayer(paramsdict['in_features'], paramsdict['out_features'], weight, biastensor)
		eps = epstensor.item()
		X = input_.clone().detach().requires_grad_(True)
		R = lrp_backward(_input=X, layer=module, relevance_output=grad_output[0], eps0=eps, eps=eps)
		return R, None, None


class posnegconv_tensorbiased(nn.Module):

	def _clone_module(self, module):
		clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
						  **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
		return clone.to(module.weight.device)

	def __init__(self, tensorbiasedconv, ignorebias):
		super(posnegconv_tensorbiased, self).__init__()

		self.posconv = tensorbiased_convlayer(tensorbiasedconv.conv.weight, tensorbiasedconv.baseconv,
											  tensorbiasedconv.inputfornewbias)
		self.negconv = tensorbiased_convlayer(tensorbiasedconv.conv.weight, tensorbiasedconv.baseconv,
											  tensorbiasedconv.inputfornewbias)

		self.posconv.conv.weight = torch.nn.Parameter(tensorbiasedconv.conv.weight.data.clone().clamp(min=0)).to(
			tensorbiasedconv.conv.weight.device)

		self.negconv.conv.weight = torch.nn.Parameter(tensorbiasedconv.conv.weight.data.clone().clamp(max=0)).to(
			tensorbiasedconv.conv.weight.device)

		if ignorebias:
			self.posconv.inputfornewbias = None
			self.negconv.inputfornewbias = None
		else:
			self.posconv.biasmode = 'pos'
			self.negconv.biasmode = 'neg'

	def forward(self, x):
		vp = self.posconv(torch.clamp(x, min=0))
		vn = self.negconv(torch.clamp(x, max=0))
		return vp + vn


class tensorbiasedconv2d_beta0_wrapper_fct(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, tensorbiasedclass, lrpignorebias):
		def configvalues_totensorlist(conv2dclass):
			propertynames = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
			values = []
			for attr in propertynames:
				v = getattr(conv2dclass, attr)
				# convert it into tensor
				# has no treatment for booleans yet
				if isinstance(v, int):
					v = torch.tensor([v], dtype=torch.int32, device=conv2dclass.weight.device)
				elif isinstance(v, tuple):
					# FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
					v = torch.tensor(v, dtype=torch.int32, device=conv2dclass.weight.device)
				else:
					raise ValueError('Unsupported attribute', attr, type(v))
				values.append(v)
			return propertynames, values

		# stash module config params and trainable params
		propertynames, values = configvalues_totensorlist(tensorbiasedclass.baseconv)
		if tensorbiasedclass.baseconv.bias is None:
			bias = None
		else:
			bias = tensorbiasedclass.baseconv.bias.data.clone()

		if tensorbiasedclass.inputfornewbias is None:
			inputfornewbias = None
		else:
			inputfornewbias = tensorbiasedclass.inputfornewbias.data.clone()

		lrpignorebiastensor = torch.tensor([lrpignorebias], dtype=torch.bool,
										   device=tensorbiasedclass.baseconv.weight.device)
		ctx.save_for_backward(x, tensorbiasedclass.baseconv.weight.data.clone(), bias,
							  tensorbiasedclass.conv.weight.data.clone(), inputfornewbias, lrpignorebiastensor,
							  *values)  # *values unpacks the list

		return tensorbiasedclass.forward(x)

	@staticmethod
	def backward(ctx, grad_output):
		input_, conv2dweight, conv2dbias, newweight, inputfornewbias, lrpignorebiastensor, *values = ctx.saved_tensors

		# reconstruct dictionary of config parameters
		def tensorlist_todict(values):
			propertynames = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
			paramsdict = {}
			for i, n in enumerate(propertynames):
				v = values[i]
				if v.numel == 1:
					paramsdict[n] = v.item()
				else:
					alist = v.tolist()
					if len(alist) == 1:
						paramsdict[n] = alist[0]
					else:
						paramsdict[n] = tuple(alist)
			return paramsdict

		paramsdict = tensorlist_todict(values)

		if conv2dbias is None:
			conv2dclass = nn.Conv2d(**paramsdict, bias=False)
		else:
			conv2dclass = nn.Conv2d(**paramsdict, bias=True)
			conv2dclass.bias = torch.nn.Parameter(conv2dbias)

		conv2dclass.weight = torch.nn.Parameter(conv2dweight)
		tensorbiasedclass = tensorbiased_convlayer(newweight=newweight, baseconv=conv2dclass,
												   inputfornewbias=inputfornewbias)
		pnconv = posnegconv_tensorbiased(tensorbiasedclass, ignorebias=lrpignorebiastensor.item())

		X = input_.clone().detach().requires_grad_(True)
		R = lrp_backward(_input=X, layer=pnconv, relevance_output=grad_output[0], eps0=1e-12, eps=0)
		return R, None, None


######################################################
# base routines
#######################################################
def safe_divide(numerator, divisor, eps0, eps):
	return numerator / (divisor + eps0 * (divisor == 0).to(divisor) + eps * divisor.sign())


def lrp_backward(_input, layer, relevance_output, eps0, eps):
	"""
	Performs the LRP backward pass, implemented as standard forward and backward passes.
	"""
	relevance_output_data = relevance_output.clone().detach()
	with torch.enable_grad():
		Z = layer(_input)
	S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)
	Z.backward(S)
	relevance_input = _input.data * _input.grad.data
	return relevance_input


class l2_lrp(torch.autograd.Function):
	@staticmethod
	def forward(ctx, conv_features, model, pno):
		pno = torch.tensor([pno], dtype=torch.int32)
		ctx.save_for_backward(conv_features, model.prototype_vectors, pno)  # *values unpacks the list
		x2 = conv_features ** 2
		x2_patch_sum = F.conv2d(input=x2, weight=model.ones)

		p2 = model.prototype_vectors ** 2
		p2 = torch.sum(p2, dim=(1, 2, 3))
		# p2 is a vector of shape (num_prototypes,)
		# then we reshape it to (num_prototypes, 1, 1)
		p2_reshape = p2.view(-1, 1, 1)

		xp = F.conv2d(input=conv_features, weight=model.prototype_vectors)
		intermediate_result = - 2 * xp + p2_reshape  # use broadcast
		# x2_patch_sum and intermediate_result are of the same shape
		distances = F.relu(x2_patch_sum + intermediate_result)

		similarities = torch.log((distances + 1) / (distances + model.epsilon))
		return similarities

	@staticmethod
	def backward(ctx, grad_output):
		conv, prototypes, pno = ctx.saved_tensors
		pno = pno.item()
		i = conv.shape[2]
		j = conv.shape[3]
		c = conv.shape[1]
		p = prototypes.shape[0]
		prototype = prototypes[pno, :, :, :]
		prototype = prototype.repeat(1, i, j)
		conv = conv.squeeze()

		l2 = (conv - prototype) ** 2
		d = 1 / (l2 ** 2 + 1e-12)

		denom = torch.sum(d, dim=0, keepdim=True) + 1e-12
		denom = denom.repeat(c, 1, 1) + 1e-12
		R = torch.div(d, denom)

		grad_output = grad_output[:, pno, :, :].squeeze()
		grad_output = grad_output.repeat(c, 1, 1)

		R = R * grad_output
		R = torch.unsqueeze(R, dim=0)
		return R, None, None


class l2_lrp_class(torch.autograd.Function):
	@staticmethod
	def forward(ctx, conv_features, tree):
		ctx.save_for_backward(conv_features, tree.prototype_layer.prototype_vectors)
		x2 = conv_features ** 2
		x2_patch_sum = F.conv2d(input=x2, weight=tree.ones)
		p2 = tree.prototype_layer.prototype_vectors ** 2
		p2 = torch.sum(p2, dim=(1, 2, 3))
		# p2 is a vector of shape (num_prototypes,)
		# then we reshape it to (num_prototypes, 1, 1)
		p2_reshape = p2.view(-1, 1, 1)

		xp = F.conv2d(input=conv_features, weight=tree.prototype_layer.prototype_vectors)
		intermediate_result = - 2 * xp + p2_reshape  # use broadcast
		# x2_patch_sum and intermediate_result are of the same shape
		distances = F.relu(x2_patch_sum + intermediate_result)
		similarities = torch.log((distances + 1) / (distances + tree.epsilon))
		return similarities

	@staticmethod
	def backward(ctx, grad_output):
		conv, prototypes = ctx.saved_tensors
		i = conv.shape[2]
		j = conv.shape[3]
		c = conv.shape[1]
		p = prototypes.shape[0]
		## Broadcast conv to Nxsize(conv) (No. of prototypes)
		conv = conv.repeat(p, 1, 1, 1)
		prototype = prototypes.repeat(1, 1, i, j)
		conv = conv.squeeze()

		l2 = (conv - prototype) ** 2
		d = 1 / (l2 ** 2 + 1e-12)
		denom = torch.sum(d, dim=1, keepdim=True) + 1e-12
		denom = denom.repeat(1, c, 1, 1) + 1e-12
		R = torch.div(d, denom)
		grad_output = grad_output.repeat(c, 1, 1, 1)
		grad_output = grad_output.permute(1, 0, 2, 3)

		R = R * grad_output
		R = torch.sum(R, dim=0)
		R = torch.unsqueeze(R, dim=0)
		return R, None, None
