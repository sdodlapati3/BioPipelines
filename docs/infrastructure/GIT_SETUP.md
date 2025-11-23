# BioPipelines Git Remote Setup

## Repository Status

✅ **Successfully created on 3 GitHub accounts**

### Remote Repositories

1. **origin** (Primary): `SanjeevaRDodlapati/BioPipelines`
   - URL: https://github.com/SanjeevaRDodlapati/BioPipelines
   - Status: ✅ Pushed successfully (47 objects, 17.36 KiB)
   - Branch: `main` tracking `origin/main`

2. **sdodlapati3**: `sdodlapati3/BioPipelines`
   - URL: https://github.com/sdodlapati3/BioPipelines
   - Status: ⚠️ Repository created, ready for push
   
3. **sdodlapa**: `sdodlapa/BioPipelines`
   - URL: https://github.com/sdodlapa/BioPipelines
   - Status: ⚠️ Repository created, ready for push

## How to Push to Other Accounts

Each GitHub account requires separate authentication. To push to the other remotes:

### Option 1: Switch GitHub CLI Account
```bash
# Switch to sdodlapati3
gh auth switch --user sdodlapati3
git push sdodlapati3 main

# Switch to sdodlapa
gh auth switch --user sdodlapa
git push sdodlapa main

# Switch back to primary
gh auth switch --user SanjeevaRDodlapati
```

### Option 2: Use GitHub CLI for Each Push
```bash
# Push to sdodlapati3
gh auth switch --user sdodlapati3 && git push sdodlapati3 main

# Push to sdodlapa
gh auth switch --user sdodlapa && git push sdodlapa main
```

### Option 3: Push to All Remotes at Once (After Auth)
```bash
# Create a push script
cat > push_all.sh << 'EOF'
#!/bin/bash
git push origin main
gh auth switch --user sdodlapati3 && git push sdodlapati3 main
gh auth switch --user sdodlapa && git push sdodlapa main
gh auth switch --user SanjeevaRDodlapati
EOF

chmod +x push_all.sh
./push_all.sh
```

## Current Configuration

```bash
$ git remote -v

origin          https://github.com/SanjeevaRDodlapati/BioPipelines.git (fetch)
origin          https://github.com/SanjeevaRDodlapati/BioPipelines.git (push)
sdodlapa        https://github.com/sdodlapa/BioPipelines.git (fetch)
sdodlapa        https://github.com/sdodlapa/BioPipelines.git (push)
sdodlapati3     https://github.com/sdodlapati3/BioPipelines.git (fetch)
sdodlapati3     https://github.com/sdodlapati3/BioPipelines.git (push)
```

## Repository Contents

All remote repositories contain:
- 45 files
- 1,832 insertions
- 4 complete NGS pipelines (DNA-seq, RNA-seq, ChIP-seq, ATAC-seq)
- Reference download automation
- Conda environment configurations
- Documentation

**Latest Commit**: `20bf59a` - "feat: Implement RNA-seq, ChIP-seq, and ATAC-seq pipelines with GCP deployment"
