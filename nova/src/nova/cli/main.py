import click

@click.group()
def cli():
    """Nova CLI tool"""
    pass

@click.command()
def serve():
    """Start the Nova server"""
    click.echo("Serve command (not implemented)")

@click.command()
def compile():
    """Compile a Nova service"""
    click.echo("Compile command (not implemented)")

@click.command()
def build():
    """Build a Nova service"""
    click.echo("Build command (not implemented)")

@click.command()
@click.option("--target", "-t", type=click.Choice(["compoundai", "helm"]), required=True, help="Deployment target")
def deploy(target):
    """Deploy a Nova service"""
    click.echo(f"Deploying to target: {target}")

cli.add_command(serve)
cli.add_command(compile)
cli.add_command(build)
cli.add_command(deploy)

if __name__ == "__main__":
    cli()
