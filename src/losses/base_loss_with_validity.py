"""Copyright 2020 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch


class BaseLossWithValidity(object):

    def calculate_loss(self, predictions, ground_truth):
        raise NotImplementedError('Must implement BaseLossWithValidity::calculate_loss')

    def calculate_mean_loss(self, predictions, ground_truth):
        return torch.mean(self.calculate_loss(predictions, ground_truth))

    def __call__(self, predictions, gt_key, reference_dict):
        # Since we deal with sequence data, assume B x T x F (if ndim == 3)
        batch_size = predictions.shape[0]

        individual_entry_losses = []
        num_valid_entries = 0

        for b in range(batch_size):
            # Get sequence data for predictions and GT
            entry_predictions = predictions[b]
            entry_ground_truth = reference_dict[gt_key][b]

            # If validity values do not exist, return simple mean
            # NOTE: We assert for now to catch unintended errors,
            #       as we do not expect a situation where these flags do not exist.
            validity_key = gt_key + '_validity'
            assert(validity_key in reference_dict)
            # if validity_key not in reference_dict:
            #     individual_entry_losses.append(torch.mean(
            #         self.calculate_mean_loss(entry_predictions, entry_ground_truth)
            #     ))
            #     continue

            # Otherwise, we need to set invalid entries to zero
            validity = reference_dict[validity_key][b].float()
            losses = self.calculate_loss(entry_predictions, entry_ground_truth)

            # Some checks to make sure that broadcasting is not hiding errors
            # in terms of consistency in return values
            assert(validity.ndim == losses.ndim)
            assert(validity.shape[0] == losses.shape[0])

            # Make sure to scale the accumulated loss correctly
            num_valid = torch.sum(validity)
            accumulated_loss = torch.sum(validity * losses)
            if num_valid > 1:
                accumulated_loss /= num_valid
            num_valid_entries += 1
            individual_entry_losses.append(accumulated_loss)

        # Merge all loss terms to yield final single scalar
        return torch.sum(torch.stack(individual_entry_losses)) / float(num_valid_entries)
