from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
import matplotlib as mpl

mpl.use('Agg')
params = {'font.family': 'serif', 'font.serif': 'Times',
          'text.usetex': True,
          'xtick.major.size': 8,
          'ytick.major.size': 8,
          'xtick.major.width': 3,
          'ytick.major.width': 3,
          'mathtext.fontset': 'custom',
          }
mpl.rcParams.update(params)
import matplotlib.pyplot as plt


def fn_distribution_by_characteristic(fn_error_analysis, characteristic_name):
    characteristic_distribution_df = OrderedDict()
    for tidx, tiou in enumerate(fn_error_analysis.tiou_thresholds):
        matched_gt_id = fn_error_analysis.prediction[fn_error_analysis.matched_gt_id_cols[tidx]]
        unmatched_gt = fn_error_analysis.ground_truth[~fn_error_analysis.ground_truth['gt-id'].isin(matched_gt_id)]
        all_gt_value, all_gt_count = np.unique(fn_error_analysis.ground_truth[characteristic_name], return_counts=True)
        unmatched_gt_value, unmatched_gt_count = np.unique(unmatched_gt[characteristic_name], return_counts=True)

        characteristic_distribution = OrderedDict()
        sum_counts = all_gt_count.sum()
        for v, c in zip(all_gt_value, all_gt_count):
            characteristic_distribution[v] = {'all': c / sum_counts, 'unmatched': 0}
        for v, c in zip(unmatched_gt_value, unmatched_gt_count):
            characteristic_distribution[v]['unmatched'] = c / (characteristic_distribution[v]['all'] * sum_counts)

        characteristic_distribution_df[tiou] = pd.DataFrame(characteristic_distribution).T.fillna(0)

    x = list(characteristic_distribution_df.values())
    characteristic_distribution_df_mean = x[0].copy()
    for this_characteristic_distribution_df in x[1:]:
        characteristic_distribution_df_mean += this_characteristic_distribution_df
    characteristic_distribution_df['avg'] = characteristic_distribution_df_mean / len(characteristic_distribution_df)

    return characteristic_distribution_df


def fn_distribution_by_pairwaise_characteristics(fn_error_analysis, characteristic_name_1, characteristic_name_2):
    characteristic_distribution_df = OrderedDict()
    for tidx, tiou in enumerate(fn_error_analysis.tiou_thresholds):
        matched_gt_id = fn_error_analysis.prediction[fn_error_analysis.matched_gt_id_cols[tidx]]
        unmatched_gt = fn_error_analysis.ground_truth[~fn_error_analysis.ground_truth['gt-id'].isin(matched_gt_id)]
        all_gt_value, all_gt_count = [], []
        for group, this_group_gt in fn_error_analysis.ground_truth.groupby(
                [characteristic_name_1, characteristic_name_2]):
            all_gt_value.append(group)
            all_gt_count.append(len(this_group_gt))
        unmatched_gt_value, unmatched_gt_count = [], []
        for group, this_group_gt in unmatched_gt.groupby([characteristic_name_1, characteristic_name_2]):
            unmatched_gt_value.append(group)
            unmatched_gt_count.append(len(this_group_gt))
        all_gt_count = np.array(all_gt_count)
        unmatched_gt_count = np.array(unmatched_gt_count)

        characteristic_distribution = OrderedDict()
        sum_counts = all_gt_count.sum()
        for v, c in zip(all_gt_value, all_gt_count):
            characteristic_distribution[v] = {'all': c / sum_counts, 'unmatched': 0}
        for v, c in zip(unmatched_gt_value, unmatched_gt_count):
            characteristic_distribution[v]['unmatched'] = c / (characteristic_distribution[v]['all'] * sum_counts)

        characteristic_distribution_df[tiou] = pd.DataFrame(characteristic_distribution).T.fillna(0)

    x = list(characteristic_distribution_df.values())
    characteristic_distribution_df_mean = x[0].copy()
    for this_characteristic_distribution_df in x[1:]:
        characteristic_distribution_df_mean += this_characteristic_distribution_df
    characteristic_distribution_df['avg'] = characteristic_distribution_df_mean / len(characteristic_distribution_df)

    return characteristic_distribution_df


def plot_fn_analysis(fn_error_analysis, save_filename,
                     colors=['#7fc97f', '#beaed4', '#fdc086', '#386cb0', '#f0027f', '#bf5b17'],
                     characteristic_names=['context-size', 'context-distance', 'agreement', 'coverage', 'length',
                                           'num-instances'],
                     characteristic_names_in_text=['Context Size', 'Context Distance', 'Agreement', 'Coverage',
                                                   'Length', '\# Instances'],
                     characteristic_names_delta_positions=[1.25, -0.9, 0.45, 0.7, 1, -0.1],
                     buckets_order=['0', '1', '2', '3', '4', '5', '6', 'XW', 'W', 'XS', 'S', 'N', 'M', 'F', 'Inf', 'L',
                                    'XL', 'H', 'XH'],
                     figsize=(20, 3.5), fontsize=24):
    # characteristic distribution
    characteristic_distribution = OrderedDict()
    for characteristic_name in characteristic_names:
        characteristic_distribution[characteristic_name] = fn_distribution_by_characteristic(fn_error_analysis,
                                                                                             characteristic_name)

    characteristic_name_lst, bucket_lst, ratio_value_lst = [], [], []
    for characteristic_name in characteristic_names:
        values = characteristic_distribution[characteristic_name]['avg']['unmatched'].values
        xticks = characteristic_distribution[characteristic_name]['avg'].index
        for i in range(len(values)):
            characteristic_name_lst.append(characteristic_name)
            bucket_lst.append(xticks[i])
            ratio_value_lst.append(values[i])

    # characteristic-name,bucket,ratio-value
    false_negative_rate_df = pd.DataFrame({'characteristic-name': characteristic_name_lst,
                                           'bucket': bucket_lst,
                                           'ratio-value': ratio_value_lst,
                                           })
    false_negative_rate_df['order'] = pd.Categorical(false_negative_rate_df['bucket'],
                                                     categories=buckets_order, ordered=True)
    false_negative_rate_df.sort_values(by='order', inplace=True)
    false_negative_rate_df_by_characteristic_name = false_negative_rate_df.groupby('characteristic-name')

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    current_x_value = 0
    xticks_lst, xvalues_lst = [], []
    for char_idx, characteristic_name in enumerate(characteristic_names):
        this_false_negative_rate = false_negative_rate_df_by_characteristic_name.get_group(characteristic_name)
        x_values = range(current_x_value, current_x_value + len(this_false_negative_rate))
        y_values = this_false_negative_rate['ratio-value'].values * 100
        mybars = plt.bar(x_values, y_values, color=colors[char_idx])
        for bari in mybars:
            height = bari.get_height()
            plt.gca().text(bari.get_x() + bari.get_width() / 2, bari.get_height() + 0.025 * 100, '%.1f' % height,
                           ha='center', color='black', fontsize=fontsize / 1.15)
        ax.annotate(characteristic_names_in_text[char_idx],
                    xy=(current_x_value + characteristic_names_delta_positions[char_idx], 100),
                    fontsize=fontsize)

        if char_idx < len(characteristic_names) - 1:
            ax.axvline(max(x_values) + 1, linewidth=1.5, color="gray", linestyle='dotted')

        current_x_value = max(x_values) + 2
        xticks_lst.extend(this_false_negative_rate['bucket'].values.tolist())
        xvalues_lst.extend(x_values)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(size=10, direction='in', width=2)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.5)
    plt.xticks(xvalues_lst, xticks_lst, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('False Negative $(\%)$', fontsize=fontsize)
    plt.tight_layout()
    plt.ylim(0, 1.1 * 100)
    fig.savefig(save_filename, bbox_inches='tight')
    print('[Done] Output analysis is saved in %s' % save_filename)


def main(output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # THUMOS14
    colors = ['#386cb0', '#f0027f', '#bf5b17']
    characteristic_names = ['coverage', 'length', 'num-instances']
    characteristic_names_in_text = ['Coverage', 'Length', '\# Instances']
    characteristic_names_delta_positions = [0.5, 1, -0.2]
    figsize = (10, 3.5)

    my_plot_fn_analysis(save_filename=os.path.join(output_folder),
                        colors=colors,
                        characteristic_names=characteristic_names,
                        characteristic_names_in_text=characteristic_names_in_text,
                        characteristic_names_delta_positions=characteristic_names_delta_positions,
                        figsize=figsize)


def my_plot_fn_analysis(save_filename,
                        colors=['#7fc97f', '#beaed4', '#fdc086', '#386cb0', '#f0027f', '#bf5b17'],
                        characteristic_names=['context-size', 'context-distance', 'agreement', 'coverage', 'length',
                                              'num-instances'],
                        characteristic_names_in_text=['Context Size', 'Context Distance', 'Agreement', 'Coverage',
                                                      'Length', '\# Instances'],
                        characteristic_names_delta_positions=[1.25, -0.9, 0.45, 0.7, 1, -0.1],
                        buckets_order=['0', '1', '2', '3', '4', '5', '6', 'XW', 'W', 'XS', 'S', 'N', 'M', 'F', 'Inf',
                                       'L',
                                       'XL', 'H', 'XH'],
                        figsize=(20, 3.5), fontsize=24):
    # characteristic-name,bucket,ratio-value
    characteristic_name_lst = ['coverage'] * 4
    characteristic_name_lst += ['length'] * 4
    characteristic_name_lst += ['num-instances'] * 4

    bucket_lst = ['O', 'D', 'A', 'B'] * 3  # origin，decoder，actionness，both

    title = "XS"
    ratio_value_lst = [7.0, 6.7, 6.8, 6.4, 9.4, 9.2, 9.0, 8.5, 6.7, 6.7, 6.7, 6.7]

    # title = "S"
    # ratio_value_lst = [3.3, 3.7, 3.8, 4.4, 2.7, 2.1, 2.8, 3.1, 6.8, 6.3, 6.4, 6.7]

    # title = "M"
    # ratio_value_lst = [8.4, 7.2, 7.2, 9.6, 3.4, 3.7, 3.4, 4.0, 3.7, 3.9, 3.9, 3.3]

    # title = "L"
    # ratio_value_lst = [4.9, 8.2, 4.9, 6.6, 5.3, 4.4, 5.3, 5.3, 8.5, 9.7, 9.0, 8.5]

    # characteristic_name_lst = ['coverage'] * 4
    # characteristic_name_lst += ['length'] * 4
    # bucket_lst = ['O', 'D', 'A', 'B'] * 2  # origin，decoder，actionness，both
    # characteristic_names = ['coverage', 'length']
    # characteristic_names_in_text = ['Coverage', 'Length']
    # title = "XL"
    # ratio_value_lst = [6.7, 8.9, 6.7, 5.6, 16.2, 21.6, 16.2, 18.9]


    save_filename = f'{save_filename}/false_negative_analysis_{title}.pdf'
    ratio_value_lst = [x / 100 for x in ratio_value_lst]

    max_y_value = (max(ratio_value_lst) + 0.05) * 100
    max_y_value_minus = (max(ratio_value_lst) + 0.03) * 100
    false_negative_rate_df = pd.DataFrame({'characteristic-name': characteristic_name_lst,
                                           'bucket': bucket_lst,
                                           'ratio-value': ratio_value_lst,
                                           })
    false_negative_rate_df['order'] = pd.Categorical(false_negative_rate_df['bucket'],
                                                     categories=buckets_order, ordered=True)
    false_negative_rate_df.sort_values(by='order', inplace=True)
    false_negative_rate_df_by_characteristic_name = false_negative_rate_df.groupby('characteristic-name')

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    current_x_value = 0
    xticks_lst, xvalues_lst = [], []
    for char_idx, characteristic_name in enumerate(characteristic_names):
        this_false_negative_rate = false_negative_rate_df_by_characteristic_name.get_group(characteristic_name)
        x_values = range(current_x_value, current_x_value + len(this_false_negative_rate))
        y_values = this_false_negative_rate['ratio-value'].values * 100
        mybars = plt.bar(x_values, y_values, color=colors[char_idx])
        for bari in mybars:
            height = bari.get_height()
            plt.gca().text(bari.get_x() + bari.get_width() / 2, bari.get_height() + 0.005 * 100, '%.1f' % height,
                           ha='center', color='black', fontsize=fontsize / 1.15)
        ax.annotate(characteristic_names_in_text[char_idx],
                    xy=(current_x_value + characteristic_names_delta_positions[char_idx], max_y_value_minus),
                    fontsize=fontsize)

        if char_idx < len(characteristic_names) - 1:
            ax.axvline(max(x_values) + 1, linewidth=1.5, color="gray", linestyle='dotted')

        current_x_value = max(x_values) + 2
        xticks_lst.extend(this_false_negative_rate['bucket'].values.tolist())
        xvalues_lst.extend(x_values)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(size=10, direction='in', width=2)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.5)
    plt.xticks(xvalues_lst, xticks_lst, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('False Negative $(\%)$', fontsize=fontsize)
    plt.tight_layout()
    plt.ylim(0, max_y_value)

    plt.title(title, fontsize=fontsize)

    fig.savefig(save_filename, bbox_inches='tight')
    print('[Done] Output analysis is saved in %s' % save_filename)


if __name__ == '__main__':
    # python my_false_negative_analysis.py --output_folder ../output/thumos/my_false_negative_analysis
    parser = ArgumentParser(description='Run the false negative error analysis.',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output_folder', required=True, type=str,
                        help='The path to the folder in which the results will be saved')
    args = parser.parse_args()

    main(args.output_folder)
