from tbparse import SummaryReader
import os
import pandas as pd
from statistics import NormalDist
import click

@click.command()
@click.option(
    "--log_dir",
    help="Logdir containing all classifier logs in the form sampler_name_n_steps/network/seed",
    metavar="STR",
    type=str,
    required=True,
)
@click.option(
    "--save_path",
    help="Where to save csv file",
    metavar="STR",
    type=str,
    required=True,
)
def cmdline(log_dir, save_path, **opts):
    
    reader = SummaryReader(log_dir, extra_columns={'dir_name'})
    df = reader.scalars
    df = df.assign(
        sampler_name = df.dir_name.str.split("/").apply(lambda x: x[0]).str.split("_").apply(lambda x: x[0]),
        n_steps = df.dir_name.str.split("/").apply(lambda x: x[0]).str.split("_").apply(lambda x: x[1]).astype(int),
        network = df.dir_name.str.split("/").apply(lambda x: x[1]),
        seed = df.dir_name.str.split("/").apply(lambda x: x[2]).astype(int),
    )
    last_values_df = df.groupby(["sampler_name", "network", "seed", "n_steps", "tag"]).apply(lambda dt: dt.loc[dt.step.idxmax(), "value"]).reset_index()
    last_values_df.columns = last_values_df.columns.tolist()[:-1] + ["value",]

    p_ho = NormalDist(mu=0, sigma=1 / (4 * 50_000)**.5)
    last_values_df.loc[last_values_df.tag=="c2st_p_value", "tag"] = "c2st_stat"
    last_values_df.loc[last_values_df.tag=="c2st_stat", "value"] += 0.5
    to_append = last_values_df.loc[last_values_df.tag=="c2st_stat"]
    to_append["tag"] = "c2st_pvalue"
    to_append["value"] = to_append["value"].apply(lambda x: 1 - p_ho.cdf(x))
    last_values_df = pd.concat([last_values_df, to_append], axis=0)
    confidence_interval_df = last_values_df.groupby(["tag", "sampler_name", "n_steps", "network"]).value.aggregate(lambda x: f"{x.mean():.3f} ± {1.96*x.std() / (len(x)**.5):.3f}").reset_index()
    to_save = confidence_interval_df.pivot(index=["sampler_name", "n_steps", "network"], values=["value"], columns=["tag"]).reset_index()
    to_save.columns = [i[0] if i[1]=="" else i[1] for i in to_save.columns.tolist()]
    to_save.sort_values(["sampler_name", "n_steps", "network"]).to_csv(f"{save_path}/classifier_metrics.csv")
